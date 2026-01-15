
cloud(C,L,H) :- cloud_at_100m_covers(C,_,H),  L=100.
cloud(C,L,H) :- cloud_at_750m_covers(C,_,H),  L=750.
cloud(C,L,H) :- cloud_at_1_4km_covers(C,_,H), L=1400.
cloud(C,L,H) :- cloud_at_3km_covers(C,_,H),   L=3000.
cloud(C,L,H) :- cloud_at_5_5km_covers(C,_,H), L=5500.
cloud(C,L,H) :- cloud_at_9km_covers(C,_,H),   L=9000.



covered_at_hour_morning(C,H) :-
    cloud(C,L1,H),cloud(C,L2,H), L1!=L2, H>=7, H<=12.

covered_at_hour_afternoon(C,H) :-
    cloud(C,L1,H),cloud(C,L2,H), L1!=L2, H>=13, H<=19.


city_covered_at_least_morning(C,1) :-
    covered_at_hour_morning(C,H1).

city_covered_at_least_morning(C,2) :-
    covered_at_hour_morning(C,H1),
    covered_at_hour_morning(C,H2),
    H1 != H2.

city_covered_at_least_morning(C,3) :-
    covered_at_hour_morning(C,H1),
    covered_at_hour_morning(C,H2),
    covered_at_hour_morning(C,H3),
    H1 != H2, H1 != H3, H2 != H3.

city_covered_at_least_morning(C,4) :-
    covered_at_hour_morning(C,H1),
    covered_at_hour_morning(C,H2),
    covered_at_hour_morning(C,H3),
    covered_at_hour_morning(C,H4),
    H1 != H2, H1 != H3, H1 != H4,
    H2 != H3, H2 != H4,
    H3 != H4.

city_covered_at_least_morning(C,5) :-
    covered_at_hour_morning(C,H1),
    covered_at_hour_morning(C,H2),
    covered_at_hour_morning(C,H3),
    covered_at_hour_morning(C,H4),
    covered_at_hour_morning(C,H5),
    H1 != H2, H1 != H3, H1 != H4, H1 != H5,
    H2 != H3, H2 != H4, H2 != H5,
    H3 != H4, H3 != H5,
    H4 != H5.

city_covered_at_least_morning(C,6) :-
    covered_at_hour_morning(C,H1),
    covered_at_hour_morning(C,H2),
    covered_at_hour_morning(C,H3),
    covered_at_hour_morning(C,H4),
    covered_at_hour_morning(C,H5),
    covered_at_hour_morning(C,H6),
    H1 != H2, H1 != H3, H1 != H4, H1 != H5, H1 != H6,
    H2 != H3, H2 != H4, H2 != H5, H2 != H6,
    H3 != H4, H3 != H5, H3 != H6,
    H4 != H5, H4 != H6,
    H5 != H6.

    
city_covered_at_least_afternoon(C,1) :-
    covered_at_hour_afternoon(C,H1).

city_covered_at_least_afternoon(C,2) :-
    covered_at_hour_afternoon(C,H1),
    covered_at_hour_afternoon(C,H2),
    H1 != H2.

city_covered_at_least_afternoon(C,3) :-
    covered_at_hour_afternoon(C,H1),
    covered_at_hour_afternoon(C,H2),
    covered_at_hour_afternoon(C,H3),
    H1 != H2, H1 != H3, H2 != H3.

city_covered_at_least_afternoon(C,4) :-
    covered_at_hour_afternoon(C,H1),
    covered_at_hour_afternoon(C,H2),
    covered_at_hour_afternoon(C,H3),
    covered_at_hour_afternoon(C,H4),
    H1 != H2, H1 != H3, H1 != H4,
    H2 != H3, H2 != H4,
    H3 != H4.

city_covered_at_least_afternoon(C,5) :-
    covered_at_hour_afternoon(C,H1),
    covered_at_hour_afternoon(C,H2),
    covered_at_hour_afternoon(C,H3),
    covered_at_hour_afternoon(C,H4),
    covered_at_hour_afternoon(C,H5),
    H1 != H2, H1 != H3, H1 != H4, H1 != H5,
    H2 != H3, H2 != H4, H2 != H5,
    H3 != H4, H3 != H5,
    H4 != H5.

city_covered_at_least_afternoon(C,6) :-
    covered_at_hour_afternoon(C,H1),
    covered_at_hour_afternoon(C,H2),
    covered_at_hour_afternoon(C,H3),
    covered_at_hour_afternoon(C,H4),
    covered_at_hour_afternoon(C,H5),
    covered_at_hour_afternoon(C,H6),
    H1 != H2, H1 != H3, H1 != H4, H1 != H5, H1 != H6,
    H2 != H3, H2 != H4, H2 != H5, H2 != H6,
    H3 != H4, H3 != H5, H3 != H6,
    H4 != H5, H4 != H6,
    H5 != H6.

% exactly 0
city_covered_exactly_morning(C,0) :-
    location(C),
    not city_covered_at_least_morning(C,1).

% exactly 1
city_covered_exactly_morning(C,1) :-
    city_covered_at_least_morning(C,1),
    not city_covered_at_least_morning(C,2).

% exactly 2
city_covered_exactly_morning(C,2) :-
    city_covered_at_least_morning(C,2),
    not city_covered_at_least_morning(C,3).

% exactly 3
city_covered_exactly_morning(C,3) :-
    city_covered_at_least_morning(C,3),
    not city_covered_at_least_morning(C,4).

% exactly 4
city_covered_exactly_morning(C,4) :-
    city_covered_at_least_morning(C,4),
    not city_covered_at_least_morning(C,5).

% exactly 5
city_covered_exactly_morning(C,5) :-
    city_covered_at_least_morning(C,5),
    not city_covered_at_least_morning(C,6).

% exactly 6 (maximum)
city_covered_exactly_morning(C,6) :-
    city_covered_at_least_morning(C,6).

% exactly 0
city_covered_exactly_afternoon(C,0) :-
    location(C),
    not city_covered_at_least_afternoon(C,1).

% exactly 1
city_covered_exactly_afternoon(C,1) :-
    city_covered_at_least_afternoon(C,1),
    not city_covered_at_least_afternoon(C,2).

% exactly 2
city_covered_exactly_afternoon(C,2) :-
    city_covered_at_least_afternoon(C,2),
    not city_covered_at_least_afternoon(C,3).

% exactly 3
city_covered_exactly_afternoon(C,3) :-
    city_covered_at_least_afternoon(C,3),
    not city_covered_at_least_afternoon(C,4).

% exactly 4
city_covered_exactly_afternoon(C,4) :-
    city_covered_at_least_afternoon(C,4),
    not city_covered_at_least_afternoon(C,5).

% exactly 5
city_covered_exactly_afternoon(C,5) :-
    city_covered_at_least_afternoon(C,5),
    not city_covered_at_least_afternoon(C,6).

% exactly 6 (maximum)
city_covered_exactly_afternoon(C,6) :-
    city_covered_at_least_afternoon(C,6).

city_clear_morning(C) :-
    location(C),
    not city_covered_at_least_morning(C,1).

city_clear_afternoon(C) :-
    location(C),
    not city_covered_at_least_afternoon(C,1).



%----------------------------------------------------

city_covered_less_than(C,1) :-
    city_clear_morning(C),
    city_clear_afternoon(C),
    location(C). 

city_covered_less_than(C,2) :-
    city_clear_morning(C),
    city_clear_afternoon(C),
    location(C).

city_covered_less_than(C,2) :-
    city_covered_exactly_morning(C,1),
    city_clear_afternoon(C),
    location(C).

city_covered_less_than(C,2) :-
    city_clear_morning(C),
    city_covered_exactly_afternoon(C,1),
    location(C).

city_covered_less_than(C,3) :-
    city_clear_morning(C),
    city_clear_afternoon(C),
    location(C).

city_covered_less_than(C,3) :-
    city_covered_exactly_morning(C,1),
    city_clear_afternoon(C),
    location(C).

city_covered_less_than(C,3) :-
    city_clear_morning(C),
    city_covered_exactly_afternoon(C,1),
    location(C).

city_covered_less_than(C,3) :-
    city_covered_exactly_morning(C,1),
    city_covered_exactly_afternoon(C,1),
    location(C).

city_covered_less_than(C,3) :-
    city_covered_exactly_morning(C,2),
    city_clear_afternoon(C),
    location(C).

city_covered_less_than(C,3) :-
    city_clear_morning(C),
    city_covered_exactly_afternoon(C,2),
    location(C).

city_covered_less_than(C,4) :-
    city_clear_morning(C),
    city_clear_afternoon(C),
    location(C).

city_covered_less_than(C,4) :-
    city_covered_exactly_morning(C,1),
    city_clear_afternoon(C),
    location(C).

city_covered_less_than(C,4) :-
    city_clear_morning(C),
    city_covered_exactly_afternoon(C,1),
    location(C).

city_covered_less_than(C,4) :-
    city_covered_exactly_morning(C,1),
    city_covered_exactly_afternoon(C,1),
    location(C).

city_covered_less_than(C,4) :-
    city_covered_exactly_morning(C,2),
    city_clear_afternoon(C),
    location(C).

city_covered_less_than(C,4) :-
    city_clear_morning(C),
    city_covered_exactly_afternoon(C,2),
    location(C).

city_covered_less_than(C,4) :-
    city_covered_exactly_morning(C,2),
    city_covered_exactly_afternoon(C,1),
    location(C).

city_covered_less_than(C,4) :-
    city_covered_exactly_morning(C,1),
    city_covered_exactly_afternoon(C,2),
    location(C).

city_covered_less_than(C,4) :-
    city_covered_exactly_morning(C,3),
    city_clear_afternoon(C),
    location(C).

city_covered_less_than(C,4) :-
    city_clear_morning(C),
    city_covered_exactly_afternoon(C,3),
    location(C).

city_covered_less_than(C,5) :-
    city_clear_morning(C), city_clear_afternoon(C), location(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,1), city_clear_afternoon(C), location(C).

city_covered_less_than(C,5) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,1), location(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,1), location(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,2), city_clear_afternoon(C), location(C).

city_covered_less_than(C,5) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,2), location(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,2), location(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,2), city_covered_exactly_afternoon(C,1), location(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,3), city_clear_afternoon(C), location(C).

city_covered_less_than(C,5) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,3), location(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,3), location(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,3), city_covered_exactly_afternoon(C,1), location(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,2), city_covered_exactly_afternoon(C,2), location(C).

city_covered_less_than(C,5) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,4), location(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,4), city_clear_afternoon(C), location(C).


city_covered_less_than(C,6) :-
    city_clear_morning(C), city_clear_afternoon(C), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,1), city_clear_afternoon(C), location(C).

city_covered_less_than(C,6) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,1), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,1), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,2), city_clear_afternoon(C), location(C).

city_covered_less_than(C,6) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,2), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,2), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,2), city_covered_exactly_afternoon(C,1), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,3), city_clear_afternoon(C), location(C).

city_covered_less_than(C,6) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,3), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,3), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,3), city_covered_exactly_afternoon(C,1), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,2), city_covered_exactly_afternoon(C,2), location(C).

city_covered_less_than(C,6) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,4), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,4), city_clear_afternoon(C), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,4), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,4), city_covered_exactly_afternoon(C,1), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,2), city_covered_exactly_afternoon(C,3), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,3), city_covered_exactly_afternoon(C,2), location(C).

city_covered_less_than(C,6) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,5), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,5), city_clear_afternoon(C), location(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,5), location(C).

temperature_increased_at_afternoon(C) :- humidity_at_afternoon(C,X), humidity_at_morning(C,Y), X>Y, location(C).
temperature_decreased_at_afternoon(C) :- humidity_at_afternoon(C,X), humidity_at_morning(C,Y), X<Y, location(C).
humidity_increased_at_afternoon(C) :- humidity_at_afternoon(C,X), humidity_at_morning(C,Y), X>Y, location(C).
humidity_decreased_at_afternoon(C) :- humidity_at_afternoon(C,X), humidity_at_morning(C,Y), X<Y, location(C).

time(0..23).                    
lv(1..6).

               
location(sappada_forni_villa).
location(pontebba_tarvisio).
location(lignano_grado).
location(barcis).
location(udine_palamnova).
location(gorizia).
location(trieste).
location(gemona_stolvizza).
location(pordenone).


coverage("cloudy"). 
coverage("partly_cloudy"). 
coverage("mostly_cloudy"). 
coverage("mostly_clear"). 
coverage("sunny"). 



%CLOUD COVER (if)
sunny_at(X) :- forecasted_sky(X, "sunny").
sunny_at(X) :- forecasted_sky(X, "mostly_clear").
partially_sunny_at(X) :- forecasted_sky(X, "partly_cloudy").
covered_at(X) :- forecasted_sky(X, "mostly_cloudy").
covered_at(X) :- forecasted_sky(X, "cloudy").

%other implication verse
:- sunny_at(X), not forecasted_sky(X, "sunny"), not forecasted_sky(X, "mostly_clear").
:- partially_sunny_at(X), not forecasted_sky(X, "partly_cloudy").
:- covered_at(X), not forecasted_sky(X, "mostly_cloudy"), not forecasted_sky(X, "cloudy").

%only one is true
:- sunny_at(X), partially_sunny_at(X).
:- sunny_at(X), covered_at(X).
:- partially_sunny_at(X), covered_at(X).

#maxv(1).


#modeh(forecasted_sky(const(location),const(coverage))).
#modeb(city_covered_at_least_morning(const(location),const(lv))).
#modeb(city_covered_at_least_afternoon(const(location),const(lv))).
#modeb(city_covered_at_least(const(location),const(lv))).
#modeb(city_covered_less_than(const(location),const(lv))).
#modeb(city_covered_exactly(const(location),const(lv))).
#modeb(not city_covered_at_least_morning(const(location),const(lv))).
#modeb(not city_covered_at_least_afternoon(const(location),const(lv))).
#modeb(temperature_increased_at_afternoon(const(location))).
#modeb(temperature_decreased_at_afternoon(const(location))).
#modeb(humidity_increased_at_afternoon(const(location))).
#modeb(humidity_decreased_at_afternoon(const(location))).

#bias("penalty(1, body(X)) :- in_body(X).").

