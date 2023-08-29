FROM neurips23

RUN apt update
RUN apt install git
RUN apt install python-is-python3
RUN git clone https://github.com/hhy3/pyglass --branch test_object
WORKDIR /home/app/pyglass
RUN pip install python/libs/*
WORKDIR /home/app

