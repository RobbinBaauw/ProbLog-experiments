%%% -*- Mode: Prolog; -*-

% definition of acyclic path using list of visited nodes
path(X,Y) :- path(X,Y,[X],_).

path(X,X,A,A).
path(X,Y,A,R) :-
	X\==Y,
	edge(X,Z),
	absent(Z,A),
	path(Z,Y,[Z|A],R).

% using directed edges in both directions
% edge(X,Y) :- dir_edge(Y,X).
edge(X,Y) :- dir_edge(X,Y).

% checking whether node hasn't been visited before
absent(_,[]).
absent(X,[Y|Z]):-X \= Y, absent(X,Z).

:- consult(edges).
