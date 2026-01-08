% Example generated data for day (2019, 11, 10)

#pos(e1,{ 

% date(2019,11,10),

forecasted_sky(sappada_forni_villa, "sunny"),
forecasted_sky(pontebba_tarvisio, "mostly_clear"),
forecasted_sky(lignano_grado, "sunny"),
forecasted_sky(barcis, "sunny"),
forecasted_sky(udine_palamnova, "sunny"),
forecasted_sky(gorizia, "sunny"),
forecasted_sky(trieste, "sunny"),
forecasted_sky(gemona_stolvizza, "sunny"),
forecasted_sky(pordenone, "sunny")
},
{
partially_sunny_at(sappada_forni_villa), 
covered_at(sappada_forni_villa), 
partially_sunny_at(pontebba_tarvisio), 
covered_at(pontebba_tarvisio), 
partially_sunny_at(lignano_grado), 
covered_at(lignano_grado), 
partially_sunny_at(barcis), 
covered_at(barcis), 
partially_sunny_at(udine_palamnova), 
covered_at(udine_palamnova), 
partially_sunny_at(gorizia), 
covered_at(gorizia), 
partially_sunny_at(trieste), 
covered_at(trieste), 
partially_sunny_at(gemona_stolvizza), 
covered_at(gemona_stolvizza), 
partially_sunny_at(pordenone), 
covered_at(pordenone)
},
{
% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)


% Humidity front data:
% humidty_front(location_1,location_2,hh): between the two locations there's a sharp change 
}). 

%general bg rules

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

covered_at_hour(C,H) :-
    cloud(C,L1,H),
    cloud(C,L2,H),
    L1 != L2.

city_covered_at_least(C,2) :-
    covered_hour(C,H1),
    covered_hour(C,H2),
    H1 != H2.

city_covered_at_least(C,3) :-
    covered_hour(C,H1),
    covered_hour(C,H2),
    covered_hour(C,H3),
    H1 != H2, H1 != H3, H2 != H3.

city_covered_at_least(C,4) :-
    covered_at_hour(C,H1),
    covered_at_hour(C,H2),
    covered_at_hour(C,H3),
    covered_at_hour(C,H4),
    H1 != H2, H1 != H3, H1 != H4,
    H2 != H3, H2 != H4,
    H3 != H4.

city_covered_at_least(C,5) :-
    covered_at_hour(C,H1),
    covered_at_hour(C,H2),
    covered_at_hour(C,H3),
    covered_at_hour(C,H4),
    covered_at_hour(C,H5),
    H1 != H2, H1 != H3, H1 != H4, H1 != H5,
    H2 != H3, H2 != H4, H2 != H5,
    H3 != H4, H3 != H5,
    H4 != H5.

city_covered_at_least(C,6) :-
    covered_at_hour(C,H1),
    covered_at_hour(C,H2),
    covered_at_hour(C,H3),
    covered_at_hour(C,H4),
    covered_at_hour(C,H5),
    covered_at_hour(C,H6),
    H1 != H2, H1 != H3, H1 != H4, H1 != H5, H1 != H6,
    H2 != H3, H2 != H4, H2 != H5, H2 != H6,
    H3 != H4, H3 != H5, H3 != H6,
    H4 != H5, H4 != H6,
    H5 != H6.

city_covered_at_least(C,7) :-
    covered_at_hour(C,H1),
    covered_at_hour(C,H2),
    covered_at_hour(C,H3),
    covered_at_hour(C,H4),
    covered_at_hour(C,H5),
    covered_at_hour(C,H6),
    covered_at_hour(C,H7),
    H1 != H2, H1 != H3, H1 != H4, H1 != H5, H1 != H6, H1 != H7,
    H2 != H3, H2 != H4, H2 != H5, H2 != H6, H2 != H7,
    H3 != H4, H3 != H5, H3 != H6, H3 != H7,
    H4 != H5, H4 != H6, H4 != H7,
    H5 != H6, H5 != H7,
    H6 != H7.



city_covered_at_least(C,8) :-
    covered_at_hour(C,H1),
    covered_at_hour(C,H2),
    covered_at_hour(C,H3),
    covered_at_hour(C,H4),
    covered_at_hour(C,H5),
    covered_at_hour(C,H6),
    covered_at_hour(C,H7),
    covered_at_hour(C,H8),

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
#modeb(city_covered_at_least(var(location),2)).
#modeb(city_covered_at_least(var(location),3)).
#modeb(city_covered_at_least(var(location),8)).

#modeb(not city_covered_at_least(var(location),2)).
#modeb(not city_covered_at_least(var(location),3)).
#modeb(not city_covered_at_least(var(location),8)).

                    