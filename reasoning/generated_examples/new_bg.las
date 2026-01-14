% Example generated data for day (2019, 11, 1)

cloud(C,L,H) :- cloud_at_100m_covers(C,_,H),  L=100.
cloud(C,L,H) :- cloud_at_750m_covers(C,_,H),  L=750.
cloud(C,L,H) :- cloud_at_1_4km_covers(C,_,H), L=1400.
cloud(C,L,H) :- cloud_at_3km_covers(C,_,H),   L=3000.
cloud(C,L,H) :- cloud_at_5_5km_covers(C,_,H), L=5500.
cloud(C,L,H) :- cloud_at_9km_covers(C,_,H),   L=9000.



covered_at_hour(C,H) :-
    cloud(C,L1,H).

city_covered_at_least(C,1) :-
    covered_at_hour(C,H1).

city_covered_at_least(C,2) :-
    covered_at_hour(C,H1),
    covered_at_hour(C,H2),
    H1 != H2.

city_covered_at_least(C,3) :-
    covered_at_hour(C,H1),
    covered_at_hour(C,H2),
    covered_at_hour(C,H3),
    H1 != H2, H1 != H3, H2 != H3.

city_covered_at_least(C,4) :-
    covered_at_hour(C,H1),
    covered_at_hour(C,H2),
    covered_at_hour(C,H3),
    covered_at_hour(C,H4),
    H1 != H2, H1 != H3, H1 != H4,
    H2 != H3, H2 != H4,
    H3 != H4.

city_clear_at_least(C,5) :-
    clear_at_hour(C,H1),
    clear_at_hour(C,H2),
    clear_at_hour(C,H3),
    clear_at_hour(C,H4),
    clear_at_hour(C,H5),

    H1 != H2, H1 != H3, H1 != H4, H1 != H5,
    H2 != H3, H2 != H4, H2 != H5,
    H3 != H4, H3 != H5,
    H4 != H5.

city_clear_at_least(C,6) :-
    clear_at_hour(C,H1),
    clear_at_hour(C,H2),
    clear_at_hour(C,H3),
    clear_at_hour(C,H4),
    clear_at_hour(C,H5),
    clear_at_hour(C,H6),

    H1 != H2, H1 != H3, H1 != H4, H1 != H5, H1 != H6,
    H2 != H3, H2 != H4, H2 != H5, H2 != H6,
    H3 != H4, H3 != H5, H3 != H6,
    H4 != H5, H4 != H6,
    H5 != H6.

city_not_covered_more_than(C,6) :- not city_covered_at_least(C,6), location(C). %placeholder
city_not_covered_more_than(C,5) :- not city_covered_at_least(C,6), location(C). 
city_not_covered_more_than(C,4) :- not city_covered_at_least(C,5), location(C).
city_not_covered_more_than(C,3) :- not city_covered_at_least(C,4), location(C).
city_not_covered_more_than(C,2) :- not city_covered_at_least(C,3), location(C).
city_not_covered_more_than(C,1) :- not city_covered_at_least(C,2), location(C).

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
#modeh(forecasted_sky(var(location),const(coverage))).

#modeh(forecasted_sky(const(location),const(coverage))).
#modeb(city_covered_at_least(var(location),const(lv))).
#modeb(city_not_covered_more_than(var(location),const(lv))).

#modeb(city_covered_at_least(const(location),const(lv))).
#modeb(city_not_covered_more_than(const(location),const(lv))).

%#modeb(city_covered_at_least(const(location),num_var(lv))).
%#modeb(city_not_covered_more_than(const(location),num_var(lv))).
