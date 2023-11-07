FROM ubuntu:14.04

RUN apt-get update && apt-get install -y git wget libboost-all-dev build-essential libncurses5-dev libreadline6-dev libgmp3-dev time

RUN cd /opt && git clone https://github.com/vscosta/yap-6.3.git && cd ./yap-6.3 && git checkout ed0d3f6cae1c608c2362189acf53a647279eca40
RUN cd /opt && wget https://src.fedoraproject.org/repo/pkgs/cudd/cudd-2.4.1.tar.gz/38f4dc5195a746222f1e51c459b49b4f/cudd-2.4.1.tar.gz && tar -xf cudd-2.4.1.tar.gz && rm cudd-2.4.1.tar.gz

RUN cd /opt/cudd-2.4.1 && sed -i 's/-mcpu=pentiumpro//g' Makefile && make
RUN cd /opt/yap-6.3 && ./configure --with-cudd="/opt/cudd-2.4.1" --enable-tabling=yes --with-readline=no && make && make install
