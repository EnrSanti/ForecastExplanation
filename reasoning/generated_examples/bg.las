

%RAINS
%rains_at(X) :- forecasted_rain(X, Y), Y > 0.
%:- rains_at(X), forecasted_rain(X, 0).   % constraint for “only if”

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



cloud(C,L,H) :- cloud_at_100m_covers(C,_,H),  L=100.
cloud(C,L,H) :- cloud_at_750m_covers(C,_,H),  L=750.
cloud(C,L,H) :- cloud_at_1_4km_covers(C,_,H), L=1400.
cloud(C,L,H) :- cloud_at_3km_covers(C,_,H),   L=3000.
cloud(C,L,H) :- cloud_at_5_5km_covers(C,_,H), L=5500.
cloud(C,L,H) :- cloud_at_9km_covers(C,_,H),   L=9000.


clear_at_hour(C,H) :-
    sun_hour(H),
    location(C),
    not cloud(C,_,H).

city_clear_at_least(C,1) :-
    clear_at_hour(C,H).

% >= 2 hours of sun
city_clear_at_least(C,2) :-
    clear_at_hour(C,H1).

city_clear_at_least(C,3) :-
    clear_at_hour(C,H1),
    clear_at_hour(C,H2),
    clear_at_hour(C,H3),
    H1 != H2, H1 != H3, H2 != H3.

city_clear_at_least(C,4) :-
    clear_at_hour(C,H1),
    clear_at_hour(C,H2),
    clear_at_hour(C,H3),
    clear_at_hour(C,H4),
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

city_clear_at_least(C,7) :-
    clear_at_hour(C,H1),
    clear_at_hour(C,H2),
    clear_at_hour(C,H3),
    clear_at_hour(C,H4),
    clear_at_hour(C,H5),
    clear_at_hour(C,H6),
    clear_at_hour(C,H7),

    H1 != H2, H1 != H3, H1 != H4, H1 != H5, H1 != H6, H1 != H7,
    H2 != H3, H2 != H4, H2 != H5, H2 != H6, H2 != H7,
    H3 != H4, H3 != H5, H3 != H6, H3 != H7,
    H4 != H5, H4 != H6, H4 != H7,
    H5 != H6, H5 != H7,
    H6 != H7.

city_clear_at_least(C,8) :-
    clear_at_hour(C,H1),
    clear_at_hour(C,H2),
    clear_at_hour(C,H3),
    clear_at_hour(C,H4),
    clear_at_hour(C,H5),
    clear_at_hour(C,H6),
    clear_at_hour(C,H7),
    clear_at_hour(C,H8),

    H1 != H2, H1 != H3, H1 != H4, H1 != H5, H1 != H6, H1 != H7, H1 != H8,
    H2 != H3, H2 != H4, H2 != H5, H2 != H6, H2 != H7, H2 != H8,
    H3 != H4, H3 != H5, H3 != H6, H3 != H7, H3 != H8,
    H4 != H5, H4 != H6, H4 != H7, H4 != H8,
    H5 != H6, H5 != H7, H5 != H8,
    H6 != H7, H6 != H8,
    H7 != H8.


city_not_covered_more_than(C,8) :- not city_covered_at_least(C,8), location(C). %placeholder
city_not_covered_more_than(C,7) :- not city_covered_at_least(C,8), location(C). 
city_not_covered_more_than(C,6) :- not city_covered_at_least(C,7), location(C). 
city_not_covered_more_than(C,5) :- not city_covered_at_least(C,6), location(C). 
city_not_covered_more_than(C,4) :- not city_covered_at_least(C,5), location(C).
city_not_covered_more_than(C,3) :- not city_covered_at_least(C,4), location(C).
city_not_covered_more_than(C,2) :- not city_covered_at_least(C,3), location(C).
city_not_covered_more_than(C,1) :- not city_covered_at_least(C,2), location(C).

time(0..23).
lv(1..8).
                    
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



is_winter(date(Y,M,D)) :-
    date(Y,M,D),
    M = 12.

is_winter(date(Y,M,D)) :-
    date(Y,M,D),
    M = 1.

is_winter(date(Y,M,D)) :-
    date(Y,M,D),
    M = 2.

is_summer(date(Y,M,D)) :-
    date(Y,M,D),
    M >= 6,
    M <= 8.

is_spring(date(Y,M,D)) :-
    date(Y,M,D),
    M >= 3,
    M <= 5.

is_autumn(date(Y,M,D)) :-
    date(Y,M,D),
    M >= 9,
    M <= 11.
    

sun_hour(H) :- time(H), is_autumn(date(Y,M,D)), H >= 6, H <= 17.
sun_hour(H) :- time(H), is_winter(date(Y,M,D)), H >= 8, H <= 16.
sun_hour(H) :- time(H), is_summer(date(Y,M,D)), H >= 5, H <= 21.
sun_hour(H) :- time(H), is_spring(date(Y,M,D)), H >= 6, H <= 19.


#modeh(forecasted_sky(var(location),const(coverage))).
#modeb(city_covered_at_least(var(location),const(lv))).
#modeb(city_not_covered_more_than(var(location),const(lv))).
#bias("penalty(1, body(X)) :- in_body(X).").

    