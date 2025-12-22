
%general bg rules

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


covered_at_hour_single_lv(C,H) :-
    cloud(C,L1,H).

city_covered_at_least_single(C,2) :-
    covered_at_hour_single_lv(C,H1),
    covered_at_hour_single_lv(C,H2),
    H1 != H2.

city_covered_at_least_single(C,3) :-
    covered_at_hour_single_lv(C,H1),
    covered_at_hour_single_lv(C,H2),
    covered_at_hour_single_lv(C,H3),
    H1 != H2, H1 != H3, H2 != H3.

city_covered_at_least_single(C,4) :-
    covered_at_hour_single_lv(C,H1),
    covered_at_hour_single_lv(C,H2),
    covered_at_hour_single_lv(C,H3),
    covered_at_hour_single_lv(C,H4),
    H1 != H2, H1 != H3, H1 != H4,
    H2 != H3, H2 != H4,
    H3 != H4.


city_covered_at_least_single(C,5) :-
    covered_at_hour_single_lv(C,H1),
    covered_at_hour_single_lv(C,H2),
    covered_at_hour_single_lv(C,H3),
    covered_at_hour_single_lv(C,H4),
    covered_at_hour_single_lv(C,H5),
    H1 != H2, H1 != H3, H1 != H4, H1 != H5,
    H2 != H3, H2 != H4, H2 != H5,
    H3 != H4, H3 != H5,
    H4 != H5.

city_covered_at_least_single(C,5) :-
    covered_at_hour_single_lv(C,H1),
    covered_at_hour_single_lv(C,H2),
    covered_at_hour_single_lv(C,H3),
    covered_at_hour_single_lv(C,H4),
    covered_at_hour_single_lv(C,H5),
    H1 != H2, H1 != H3, H1 != H4, H1 != H5,
    H2 != H3, H2 != H4, H2 != H5,
    H3 != H4, H3 != H5,
    H4 != H5.

city_covered_at_least_single(C,6) :-
    covered_at_hour_single_lv(C,H1),
    covered_at_hour_single_lv(C,H2),
    covered_at_hour_single_lv(C,H3),
    covered_at_hour_single_lv(C,H4),
    covered_at_hour_single_lv(C,H5),
    covered_at_hour_single_lv(C,H6),
    H1 != H2, H1 != H3, H1 != H4, H1 != H5, H1 != H6,
    H2 != H3, H2 != H4, H2 != H5, H2 != H6,
    H3 != H4, H3 != H5, H3 != H6,
    H4 != H5, H4 != H6,
    H5 != H6.


city_covered_at_least_single(C,7) :-
    covered_at_hour_single_lv(C,H1),
    covered_at_hour_single_lv(C,H2),
    covered_at_hour_single_lv(C,H3),
    covered_at_hour_single_lv(C,H4),
    covered_at_hour_single_lv(C,H5),
    covered_at_hour_single_lv(C,H6),
    covered_at_hour_single_lv(C,H7),
    H1 != H2, H1 != H3, H1 != H4, H1 != H5, H1 != H6, H1 != H7,
    H2 != H3, H2 != H4, H2 != H5, H2 != H6, H2 != H7,
    H3 != H4, H3 != H5, H3 != H6, H3 != H7,
    H4 != H5, H4 != H6, H4 != H7,
    H5 != H6, H5 != H7,
    H6 != H7.

city_covered_at_least_single(C,8) :-
    covered_at_hour_single_lv(C,H1),
    covered_at_hour_single_lv(C,H2),
    covered_at_hour_single_lv(C,H3),
    covered_at_hour_single_lv(C,H4),
    covered_at_hour_single_lv(C,H5),
    covered_at_hour_single_lv(C,H6),
    covered_at_hour_single_lv(C,H7),
    covered_at_hour_single_lv(C,H8),

    H1 != H2, H1 != H3, H1 != H4, H1 != H5, H1 != H6, H1 != H7, H1 != H8,
    H2 != H3, H2 != H4, H2 != H5, H2 != H6, H2 != H7, H2 != H8,
    H3 != H4, H3 != H5, H3 != H6, H3 != H7, H3 != H8,
    H4 != H5, H4 != H6, H4 != H7, H4 != H8,
    H5 != H6, H5 != H7, H5 != H8,
    H6 != H7, H6 != H8,
    H7 != H8.

time(0..23).
                    
location(sappada_forni_villa).
location(pontebba_tarvisio).
location(lignano_grado).
location(barcis).
location(udine_palamnova).
location(gorizia).
location(trieste).
location(gemona_stolvizza).
location(pordenone).

coverage("mostly_cloudy").
coverage("partly_cloudy").
coverage("small_cloud").
coverage("mostly_clear").
coverage("cloud").
coverage("cloudy").
coverage("sunny").
%coverage("ND").

#maxv(3).
#modeh(forecasted_sky(var(location),var(coverage))).
#modeh(forecasted_sky(const(location),var(coverage))).
#modeh(forecasted_sky(var(location),const(coverage))).
#modeh(forecasted_sky(const(location),const(coverage))).

#modeb(city_covered_at_least(var(location),1)).
#modeb(city_covered_at_least(var(location),2)).
#modeb(city_covered_at_least(var(location),3)).
#modeb(city_covered_at_least(var(location),4)).
#modeb(city_covered_at_least(var(location),5)).
#modeb(city_covered_at_least(var(location),6)).
#modeb(city_covered_at_least(var(location),7)).
#modeb(city_covered_at_least(var(location),8)).

#modeb(not city_covered_at_least(var(location),1)).
#modeb(not city_covered_at_least(var(location),2)).
#modeb(not city_covered_at_least(var(location),3)).
#modeb(not city_covered_at_least(var(location),4)).
#modeb(not city_covered_at_least(var(location),5)).
#modeb(not city_covered_at_least(var(location),6)).
#modeb(not city_covered_at_least(var(location),7)).
#modeb(not city_covered_at_least(var(location),8)).

#modeb(city_covered_at_least(const(location),1)).
#modeb(city_covered_at_least(const(location),2)).
#modeb(city_covered_at_least(const(location),3)).
#modeb(city_covered_at_least(const(location),4)).
#modeb(city_covered_at_least(const(location),5)).
#modeb(city_covered_at_least(const(location),6)).
#modeb(city_covered_at_least(const(location),7)).
#modeb(city_covered_at_least(const(location),8)).

#modeb(not city_covered_at_least(const(location),1)).
#modeb(not city_covered_at_least(const(location),2)).
#modeb(not city_covered_at_least(const(location),3)).
#modeb(not city_covered_at_least(const(location),4)).
#modeb(not city_covered_at_least(const(location),5)).
#modeb(not city_covered_at_least(const(location),6)).
#modeb(not city_covered_at_least(const(location),7)).
#modeb(not city_covered_at_least(const(location),8)).

                    

