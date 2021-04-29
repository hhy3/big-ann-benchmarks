import argparse
import json
import logging
import os
import threading
import time
import traceback

import colors
import docker
import numpy
import psutil

from benchmark.algorithms.definitions import (Definition,
                                               instantiate_algorithm)
from benchmark.datasets import DATASETS
from benchmark.results import store_results


def run_individual_query(algo, X, distance, count, run_count, search_type):
    best_search_time = float('inf')
    for i in range(run_count):
        print('Run %d/%d...' % (i + 1, run_count))

        start = time.time()
        if search_type == "knn":
            algo.query(X, count)
        else:
            algo.range_query(X, count)
        total = (time.time() - start)

        results = algo.get_results()

        search_time = total
        best_search_time = min(best_search_time, search_time)

    attrs = {
        "best_search_time": best_search_time,
        "name": str(algo),
        "run_count": run_count,
        "distance": distance,
        "type": search_type,
        "count": int(count)
    }
    additional = algo.get_additional()
    for k in additional:
        attrs[k] = additional[k]
    return (attrs, results)


def run(definition, dataset, count, run_count):
    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups \
           or hasattr(algo, "set_query_arguments"), """\
error: query argument groups have been specified for %s.%s(%s), but the \
algorithm instantiated from it does not implement the set_query_arguments \
function""" % (definition.module, definition.constructor, definition.arguments)

    ds = DATASETS[dataset]
    #X_train = numpy.array(D['train'])
    X =  ds.get_queries()
    distance = ds.distance()
    search_type = ds.search_type()
    print(f"Running {definition.algorithm} on {dataset}")
    print(fr"Got {len(X)} queries")

    try:
        # Try loading the index from the file
        memory_usage_before = algo.get_memory_usage()
        if not algo.load_index(dataset):
            # Build the index if it is not available
            t0 = time.time()
            algo.fit(dataset)
            build_time = time.time() - t0
            print('Built index in', build_time)

        index_size = algo.get_memory_usage() - memory_usage_before
        print('Index size: ', index_size)

        query_argument_groups = definition.query_argument_groups
        # Make sure that algorithms with no query argument groups still get run
        # once by providing them with a single, empty, harmless group
        if not query_argument_groups:
            query_argument_groups = [[]]

        for pos, query_arguments in enumerate(query_argument_groups, 1):
            print("Running query argument group %d of %d..." %
                  (pos, len(query_argument_groups)))
            if query_arguments:
                algo.set_query_arguments(*query_arguments)
            descriptor, results = run_individual_query(
                algo, X, distance, count, run_count, search_type)
            # A bit unclear how to set this correctly if we usually load from file
            #descriptor["build_time"] = build_time
            descriptor["index_size"] = index_size
            descriptor["algo"] = definition.algorithm
            descriptor["dataset"] = dataset
            store_results(dataset, count, definition,
                    query_arguments, descriptor, results)
    finally:
        algo.done()


def run_from_cmdline():
    parser = argparse.ArgumentParser('''

            NOTICE: You probably want to run.py rather than this script.

''')
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        help=f'Dataset to benchmark on.',
        required=True)
    parser.add_argument(
        '--algorithm',
        help='Name of algorithm for saving the results.',
        required=True)
    parser.add_argument(
        '--module',
        help='Python module containing algorithm. E.g. "ann_benchmarks.algorithms.annoy"',
        required=True)
    parser.add_argument(
        '--constructor',
        help='Constructer to load from module. E.g. "Annoy"',
        required=True)
    parser.add_argument(
        '--count',
        help='k: Number of nearest neighbours for the algorithm to return.',
        required=True,
        type=int)
    parser.add_argument(
        '--runs',
        help='Number of times to run the algorihm. Will use the fastest run-time over the bunch.',
        required=True,
        type=int)
    parser.add_argument(
        'build',
        help='JSON of arguments to pass to the constructor. E.g. ["angular", 100]'
        )
    parser.add_argument(
        'queries',
        help='JSON of arguments to pass to the queries. E.g. [100]',
        nargs='*',
        default=[])
    args = parser.parse_args()
    algo_args = json.loads(args.build)
    print(algo_args)
    query_args = [json.loads(q) for q in args.queries]

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None,  # not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args,
        query_argument_groups=query_args,
        disabled=False
    )
    run(definition, args.dataset, args.count, args.runs)


def run_docker(definition, dataset, count, runs, timeout, cpu_limit,
               mem_limit=None):
    cmd = ['--dataset', dataset,
           '--algorithm', definition.algorithm,
           '--module', definition.module,
           '--constructor', definition.constructor,
           '--runs', str(runs),
           '--count', str(count)]
    cmd.append(json.dumps(definition.arguments))
    cmd += [json.dumps(qag) for qag in definition.query_argument_groups]

    client = docker.from_env()
    if mem_limit is None:
        mem_limit = psutil.virtual_memory().available

    container = client.containers.run(
        definition.docker_tag,
        cmd,
        volumes={
            os.path.abspath('benchmark'):
                {'bind': '/home/app/benchmark', 'mode': 'ro'},
            os.path.abspath('data'):
                {'bind': '/home/app/data', 'mode': 'rw'},
            os.path.abspath('results'):
                {'bind': '/home/app/results', 'mode': 'rw'},
        },
        cpuset_cpus=cpu_limit,
        mem_limit=mem_limit,
        detach=True)
    logger = logging.getLogger(f"annb.{container.short_id}")

    logger.info('Created container %s: CPU limit %s, mem limit %s, timeout %d, command %s' % \
                (container.short_id, cpu_limit, mem_limit, timeout, cmd))

    def stream_logs():
        for line in container.logs(stream=True):
            logger.info(colors.color(line.decode().rstrip(), fg='blue'))

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()

    try:
        exit_code = container.wait(timeout=timeout)

        # Exit if exit code
        if exit_code not in [0, None]:
            logger.error(colors.color(container.logs().decode(), fg='red'))
            logger.error('Child process for container %s raised exception %d' % (container.short_id, exit_code))
    except:
        logger.error('Container.wait for container %s failed with exception' % container.short_id)
        logger.error('Invoked with %s' % cmd)
        traceback.print_exc()
    finally:
        container.remove(force=True)