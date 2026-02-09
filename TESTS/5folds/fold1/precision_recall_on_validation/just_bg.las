
cloud(C,L,H) :- cloud_at_100m_covers(C,_,H),  L=100.
cloud(C,L,H) :- cloud_at_750m_covers(C,_,H),  L=750.
cloud(C,L,H) :- cloud_at_1_4km_covers(C,_,H), L=1400.
cloud(C,L,H) :- cloud_at_3km_covers(C,_,H),   L=3000.
cloud(C,L,H) :- cloud_at_5_5km_covers(C,_,H), L=5500.
cloud(C,L,H) :- cloud_at_9km_covers(C,_,H),   L=9000.



possible_season(winter).
possible_season(spring).
possible_season(summer).
possible_season(autumn).

season(winter) :- date(_, MM, _), MM = 12.
season(winter) :- date(_, MM, _), MM <= 2.
season(spring) :- date(_, MM, _), MM >= 3, MM <= 5.
season(summer) :- date(_, MM, _), MM >= 6, MM <= 8.
season(autumn) :- date(_, MM, _), MM >= 9, MM <= 11.

sunlight_range(1, 8, 16).
sunlight_range(2, 7, 18).
sunlight_range(3, 6, 20).
sunlight_range(4, 7, 17).

sunlight(H) :- season(S), sunlight_range(S, From, To), time(H), H >= From, H <= To.

covered_at_hour_morning(C,H) :-
    cloud(C,L1,H), sunlight(H), time(H),location_considered(C).

covered_at_hour_afternoon(C,H) :-
    cloud(C,L1,H), sunlight(H), time(H),location_considered(C).

city_covered_at_least_morning(C,1) :-
    covered_at_hour_morning(C,H1),location_considered(C).

city_covered_at_least_morning(C,2) :-
    covered_at_hour_morning(C,H1),
    covered_at_hour_morning(C,H2),
    location_considered(C),
    H1 < H2.

city_covered_at_least_morning(C,3) :-
    covered_at_hour_morning(C,H1),
    covered_at_hour_morning(C,H2),
    covered_at_hour_morning(C,H3),
    location_considered(C),
    H1 < H2, H2 < H3.

city_covered_at_least_morning(C,4) :-
    covered_at_hour_morning(C,H1),
    covered_at_hour_morning(C,H2),
    covered_at_hour_morning(C,H3),
    covered_at_hour_morning(C,H4),
    location_considered(C),
    H1 < H2, H2 < H3, H3 < H4.

city_covered_at_least_morning(C,5) :-
    covered_at_hour_morning(C,H1),
    covered_at_hour_morning(C,H2),
    covered_at_hour_morning(C,H3),
    covered_at_hour_morning(C,H4),
    covered_at_hour_morning(C,H5),
    location_considered(C),
    H1 < H2, H2 < H3, H3 < H4, H4 < H5.


city_covered_at_least_morning(C,6) :-
    covered_at_hour_morning(C,H1),
    covered_at_hour_morning(C,H2),
    covered_at_hour_morning(C,H3),
    covered_at_hour_morning(C,H4),
    covered_at_hour_morning(C,H5),
    covered_at_hour_morning(C,H6),
    location_considered(C),
    H1 < H2, H2 < H3, H3 < H4, H4 < H5, H5 < H6.
    
city_covered_at_least_morning_(C,X) :- city_covered_at_least_morning(C,X), location_considered(C),lv(X).
city_covered_at_least_morning_neighbour(C,X) :- city_covered_at_least_morning(C,X), adjacent_to_considered(C),lv(X).


city_covered_at_least_afternoon(C,1) :-
    location_considered(C), covered_at_hour_afternoon(C,H1).

city_covered_at_least_afternoon(C,2) :-
    covered_at_hour_afternoon(C,H1),
    covered_at_hour_afternoon(C,H2),
    location_considered(C),
    H1 < H2.

city_covered_at_least_afternoon(C,3) :-
    covered_at_hour_afternoon(C,H1),
    covered_at_hour_afternoon(C,H2),
    covered_at_hour_afternoon(C,H3),
    location_considered(C),
    H1 < H2, H2 < H3.

city_covered_at_least_afternoon(C,4) :-
    covered_at_hour_afternoon(C,H1),
    covered_at_hour_afternoon(C,H2),
    covered_at_hour_afternoon(C,H3),
    covered_at_hour_afternoon(C,H4),
    location_considered(C),
    H1 < H2, H2 < H3, H3 < H4.

city_covered_at_least_afternoon(C,5) :-
    covered_at_hour_afternoon(C,H1),
    covered_at_hour_afternoon(C,H2),
    covered_at_hour_afternoon(C,H3),
    covered_at_hour_afternoon(C,H4),
    covered_at_hour_afternoon(C,H5),
    location_considered(C),
    H1 < H2, H2 < H3, H3 < H4, H4 < H5.

city_covered_at_least_afternoon(C,6) :-
    covered_at_hour_afternoon(C,H1),
    covered_at_hour_afternoon(C,H2),
    covered_at_hour_afternoon(C,H3),
    covered_at_hour_afternoon(C,H4),
    covered_at_hour_afternoon(C,H5),
    covered_at_hour_afternoon(C,H6),
    location_considered(C),
    H1 < H2, H2 < H3, H3 < H4, H4 < H5, H5 < H6.

city_covered_at_least_afternoon_(C,X) :- city_covered_at_least_afternoon(C,X), location_considered(C),lv(X).
city_covered_at_least_afternoon_neighbour(C,X) :- city_covered_at_least_afternoon(C,X), adjacent_to_considered(C),lv(X).

% exactly 0
city_covered_exactly_morning(C,0) :-
    location_considered(C),
    adjacent_to_considered(C),
    not city_covered_at_least_morning(C,1).

% exactly 1
city_covered_exactly_morning(C,1) :-
    location_considered(C),
    city_covered_at_least_morning(C,1), %already adjacent
    not city_covered_at_least_morning(C,2).

% exactly 2
city_covered_exactly_morning(C,2) :-
    location_considered(C),
    city_covered_at_least_morning(C,2), %already adjacent
    not city_covered_at_least_morning(C,3).

% exactly 3
city_covered_exactly_morning(C,3) :-
    location_considered(C),
    city_covered_at_least_morning(C,3), %already adjacent
    not city_covered_at_least_morning(C,4).

% exactly 4
city_covered_exactly_morning(C,4) :-
    location_considered(C),
    city_covered_at_least_morning(C,4), %already adjacent
    not city_covered_at_least_morning(C,5).

% exactly 5
city_covered_exactly_morning(C,5) :-
    location_considered(C),
    city_covered_at_least_morning(C,5),  %already adjacent
    not city_covered_at_least_morning(C,6).

% exactly 6 (maximum)
city_covered_exactly_morning(C,6) :-
    location_considered(C),
    city_covered_at_least_morning(C,6).  %already adjacent

% exactly 0
city_covered_exactly_afternoon(C,0) :-
    location_considered(C),
    adjacent_to_considered(C), 
    not city_covered_at_least_afternoon(C,1).

% exactly 1
city_covered_exactly_afternoon(C,1) :-
    location_considered(C),
    city_covered_at_least_afternoon(C,1), %already adjacent
    not city_covered_at_least_afternoon(C,2).

% exactly 2
city_covered_exactly_afternoon(C,2) :-
    location_considered(C),
    city_covered_at_least_afternoon(C,2), %already adjacent
    not city_covered_at_least_afternoon(C,3).

% exactly 3
city_covered_exactly_afternoon(C,3) :-
    location_considered(C),
    city_covered_at_least_afternoon(C,3), %already adjacent
    not city_covered_at_least_afternoon(C,4).

% exactly 4
city_covered_exactly_afternoon(C,4) :-
    location_considered(C),
    city_covered_at_least_afternoon(C,4), %already adjacent
    not city_covered_at_least_afternoon(C,5).

% exactly 5
city_covered_exactly_afternoon(C,5) :-
    location_considered(C),
    city_covered_at_least_afternoon(C,5), %already adjacent
    not city_covered_at_least_afternoon(C,6).

% exactly 6 (maximum)
city_covered_exactly_afternoon(C,6) :-
    location_considered(C),
    city_covered_at_least_afternoon(C,6).  %already adjacent


city_covered_exactly_afternoon_(C,X) :- city_covered_exactly_afternoon(C,X), location_considered(C),lv(X).
city_covered_exactly_afternoon_neighbour(C,X) :- city_covered_exactly_afternoon(C,X), adjacent_to_considered(C),lv(X).
city_covered_exactly_morning_(C,X) :- city_covered_exactly_morning(C,X), location_considered(C),lv(X).
city_covered_exactly_morning_neighbour(C,X) :- city_covered_exactly_morning(C,X), adjacent_to_considered(C),lv(X).

city_clear_morning(C) :-
    location_considered(C),
    not city_covered_at_least_morning(C,1).

city_clear_afternoon(C) :-
    location_considered(C),
    not city_covered_at_least_afternoon(C,1).




city_covered_less_than(C,1) :- 
    city_clear_morning(C),
    city_clear_afternoon(C),
    location_considered(C). %already adjacent

city_covered_less_than(C,2) :-
    city_clear_morning(C),
    city_clear_afternoon(C),
    location_considered(C). %already adjacent

city_covered_less_than(C,2) :-
    city_covered_exactly_morning(C,1),
    city_clear_afternoon(C),
    location_considered(C). %...

city_covered_less_than(C,2) :-
    city_clear_morning(C),
    city_covered_exactly_afternoon(C,1),
    location_considered(C).

city_covered_less_than(C,3) :-
    city_clear_morning(C),
    city_clear_afternoon(C),
    location_considered(C).

city_covered_less_than(C,3) :-
    city_covered_exactly_morning(C,1),
    city_clear_afternoon(C),
    location_considered(C).

city_covered_less_than(C,3) :-
    city_clear_morning(C),
    city_covered_exactly_afternoon(C,1),
    location_considered(C).

city_covered_less_than(C,3) :-
    city_covered_exactly_morning(C,1),
    city_covered_exactly_afternoon(C,1),
    location_considered(C).

city_covered_less_than(C,3) :-
    city_covered_exactly_morning(C,2),
    city_clear_afternoon(C),
    location_considered(C).

city_covered_less_than(C,3) :-
    city_clear_morning(C),
    city_covered_exactly_afternoon(C,2),
    location_considered(C).

city_covered_less_than(C,4) :-
    city_clear_morning(C),
    city_clear_afternoon(C),
    location_considered(C).

city_covered_less_than(C,4) :-
    city_covered_exactly_morning(C,1),
    city_clear_afternoon(C),
    location_considered(C).

city_covered_less_than(C,4) :-
    city_clear_morning(C),
    city_covered_exactly_afternoon(C,1),
    location_considered(C).

city_covered_less_than(C,4) :-
    city_covered_exactly_morning(C,1),
    city_covered_exactly_afternoon(C,1),
    location_considered(C).

city_covered_less_than(C,4) :-
    city_covered_exactly_morning(C,2),
    city_clear_afternoon(C),
    location_considered(C).

city_covered_less_than(C,4) :-
    city_clear_morning(C),
    city_covered_exactly_afternoon(C,2),
    location_considered(C).

city_covered_less_than(C,4) :-
    city_covered_exactly_morning(C,2),
    city_covered_exactly_afternoon(C,1),
    location_considered(C).

city_covered_less_than(C,4) :-
    city_covered_exactly_morning(C,1),
    city_covered_exactly_afternoon(C,2),
    location_considered(C).

city_covered_less_than(C,4) :-
    city_covered_exactly_morning(C,3),
    city_clear_afternoon(C),
    location_considered(C).

city_covered_less_than(C,4) :-
    city_clear_morning(C),
    city_covered_exactly_afternoon(C,3),
    location_considered(C).



city_covered_less_than(C,5) :-
    city_clear_morning(C), city_clear_afternoon(C), location_considered(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,1), city_clear_afternoon(C), location_considered(C).

city_covered_less_than(C,5) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,1), location_considered(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,1), location_considered(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,2), city_clear_afternoon(C), location_considered(C).

city_covered_less_than(C,5) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,2), location_considered(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,2), location_considered(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,2), city_covered_exactly_afternoon(C,1), location_considered(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,3), city_clear_afternoon(C), location_considered(C).

city_covered_less_than(C,5) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,3), location_considered(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,3), location_considered(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,3), city_covered_exactly_afternoon(C,1), location_considered(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,2), city_covered_exactly_afternoon(C,2), location_considered(C).

city_covered_less_than(C,5) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,4), location_considered(C).

city_covered_less_than(C,5) :-
    city_covered_exactly_morning(C,4), city_clear_afternoon(C), location_considered(C).


city_covered_less_than(C,6) :-
    city_clear_morning(C), city_clear_afternoon(C), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,1), city_clear_afternoon(C), location_considered(C).

city_covered_less_than(C,6) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,1), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,1), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,2), city_clear_afternoon(C), location_considered(C).

city_covered_less_than(C,6) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,2), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,2), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,2), city_covered_exactly_afternoon(C,1), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,3), city_clear_afternoon(C), location_considered(C).

city_covered_less_than(C,6) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,3), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,3), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,3), city_covered_exactly_afternoon(C,1), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,2), city_covered_exactly_afternoon(C,2), location_considered(C).

city_covered_less_than(C,6) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,4), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,4), city_clear_afternoon(C), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,4), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,4), city_covered_exactly_afternoon(C,1), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,2), city_covered_exactly_afternoon(C,3), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,3), city_covered_exactly_afternoon(C,2), location_considered(C).

city_covered_less_than(C,6) :-
    city_clear_morning(C), city_covered_exactly_afternoon(C,5), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,5), city_clear_afternoon(C), location_considered(C).

city_covered_less_than(C,6) :-
    city_covered_exactly_morning(C,1), city_covered_exactly_afternoon(C,5), location_considered(C).



city_covered_less_than_(C,X) :- city_covered_less_than(C,X), location_considered(C),lv(X).
city_covered_less_than_neighbour(C,X) :- city_covered_less_than(C,X), adjacent_to_considered(C),lv(X).



%temperature_increased_at_afternoon(C) :- humidity_at_afternoon(C,X), humidity_at_morning(C,Y), X>Y, location(C), adjacent_to_considered(C).
%temperature_decreased_at_afternoon(C) :- humidity_at_afternoon(C,X), humidity_at_morning(C,Y), X<Y, location(C), adjacent_to_considered(C).
%humidity_increased_at_afternoon(C) :- humidity_at_afternoon(C,X), humidity_at_morning(C,Y), X>Y, location(C), adjacent_to_considered(C).
%humidity_decreased_at_afternoon(C) :- humidity_at_afternoon(C,X), humidity_at_morning(C,Y), X<Y, location(C), adjacent_to_considered(C).

time(0..23).                    
lv(1..6).
wind_speed_level(0..60).


wind_dir("E").
wind_dir("NE").
wind_dir("N").
wind_dir("NW").
wind_dir("W").
wind_dir("SW").
wind_dir("S").
wind_dir("SE").


location(sappada_forni_villa).
location(pontebba_tarvisio).
location(lignano_grado).
location(barcis).
location(udine_palmanova).
location(gorizia).
location(trieste).
location(gemona_stolvizza).
location(pordenone).


coverage("cloudy"). 
coverage("partly_cloudy"). 
coverage("mostly_cloudy"). 
coverage("mostly_clear"). 
coverage("sunny"). 


sunny_at(X,S) :- forecasted_sky(X, "sunny",S),season(S),location_considered(X).
sunny_at(X,S) :- forecasted_sky(X, "mostly_clear",S),season(S),location_considered(X).
partially_sunny_at(X,S) :- forecasted_sky(X, "partly_cloudy",S),season(S),location_considered(X).
covered_at(X,S) :- forecasted_sky(X, "mostly_cloudy",S),season(S),location_considered(X).
covered_at(X,S) :- forecasted_sky(X, "cloudy",S),season(S),location_considered(X).

%other implication verse
:- sunny_at(X,S), not forecasted_sky(X, "sunny",S), not forecasted_sky(X, "mostly_clear",S),season(S),location_considered(X).
:- partially_sunny_at(X,S), not forecasted_sky(X, "partly_cloudy",S),season(S),location_considered(X).
:- covered_at(X,S), not forecasted_sky(X, "mostly_cloudy",S), not forecasted_sky(X, "cloudy",S),season(S),location_considered(X).

%only one is true
:- sunny_at(X,S), partially_sunny_at(X,S),season(S),location_considered(X).
:- sunny_at(X,S), covered_at(X,S),season(S),location_considered(X).
:- partially_sunny_at(X,S), covered_at(X,S),season(S),location_considered(X).


:- rains_at(X,S), forecasted_rain(X, 0, S), season(S),location_considered(X).

adjacent(sappada_forni_villa,pontebba_tarvisio) :- location_considered(sappada_forni_villa).
adjacent(sappada_forni_villa,gemona_stolvizza) :- location_considered(sappada_forni_villa).
adjacent(sappada_forni_villa,udine_palmanova) :- location_considered(sappada_forni_villa).
adjacent(sappada_forni_villa,pordenone) :- location_considered(sappada_forni_villa).
adjacent(sappada_forni_villa,barcis) :- location_considered(sappada_forni_villa).
adjacent(sappada_forni_villa,pontebba_tarvisio) :- location_considered(pontebba_tarvisio).
adjacent(sappada_forni_villa,gemona_stolvizza) :- location_considered(gemona_stolvizza).
adjacent(sappada_forni_villa,udine_palmanova) :- location_considered(udine_palmanova).
adjacent(sappada_forni_villa,pordenone) :- location_considered(pordenone).
adjacent(sappada_forni_villa,barcis) :- location_considered(barcis).


adjacent(pontebba_tarvisio,gemona_stolvizza) :- location_considered(pontebba_tarvisio).
adjacent(pontebba_tarvisio,gorizia) :- location_considered(pontebba_tarvisio).
adjacent(pontebba_tarvisio,gemona_stolvizza) :- location_considered(gemona_stolvizza).
adjacent(pontebba_tarvisio,gorizia) :- location_considered(gorizia).

adjacent(gemona_stolvizza,udine_palmanova) :- location_considered(gemona_stolvizza).
adjacent(gemona_stolvizza,gorizia) :- location_considered(gemona_stolvizza).
adjacent(gemona_stolvizza,udine_palmanova) :- location_considered(udine_palmanova).
adjacent(gemona_stolvizza,gorizia) :- location_considered(gorizia).

adjacent(barcis,pordenone) :- location_considered(pordenone).
adjacent(barcis,pordenone) :- location_considered(barcis).

adjacent(pordenone,udine_palmanova) :- location_considered(pordenone).
adjacent(pordenone,lignano_grado) :- location_considered(pordenone).
adjacent(pordenone,udine_palmanova) :- location_considered(udine_palmanova).
adjacent(pordenone,lignano_grado) :- location_considered(lignano_grado).

adjacent(udine_palmanova,gorizia) :- location_considered(udine_palmanova).
adjacent(udine_palmanova,lignano_grado) :- location_considered(udine_palmanova).
adjacent(udine_palmanova,gorizia) :- location_considered(gorizia).
adjacent(udine_palmanova,lignano_grado) :- location_considered(lignano_grado).

adjacent(gorizia,trieste) :- location_considered(gorizia).
adjacent(gorizia,trieste) :- location_considered(trieste).
adjacent(lignano_grado,trieste) :- location_considered(lignano_grado).
adjacent(lignano_grado,trieste) :- location_considered(trieste).

adjacent(X,X) :- location(X).
adjacent(X,Y) :- adjacent(Y,X),location(X),location(Y).

adjacent_to_considered(X) :- adjacent(X,Y), location_considered(Y),location(Y), location(X).


temperature_increased_at_afternoon_(C) :- temperature_increased_at_afternoon(C), location_considered(C).
temperature_decreased_at_afternoon_(C) :- temperature_decreased_at_afternoon(C), location_considered(C).
humidity_increased_at_afternoon_(C) :- humidity_increased_at_afternoon(C), location_considered(C).
humidity_decreased_at_afternoon_(C) :- humidity_decreased_at_afternoon(C), location_considered(C).


temperature_increased_at_afternoon_neighbour(C) :- temperature_increased_at_afternoon(C), adjacent_to_considered(C).
temperature_decreased_at_afternoon_neighbour(C) :- temperature_decreased_at_afternoon(C), adjacent_to_considered(C).
humidity_increased_at_afternoon_neighbour(C) :- humidity_increased_at_afternoon(C), adjacent_to_considered(C).
humidity_decreased_at_afternoon_neighbour(C) :- humidity_decreased_at_afternoon(C), adjacent_to_considered(C).


wind_blowing_morning_(C,DIR) :- wind_blowing_morning(C,DIR,SPEED), location_considered(C),wind_speed_level(SPEED),wind_dir(DIR).
wind_blowing_afternoon_(C,DIR) :- wind_blowing_afternoon(C,DIR,SPEED), location_considered(C),wind_speed_level(SPEED),wind_dir(DIR).

wind_blowing_morning_neighbour(C,DIR) :- wind_blowing_morning(C,DIR,SPEED), adjacent_to_considered(C),wind_speed_level(SPEED),wind_dir(DIR).
wind_blowing_afternoon_neighbour(C,DIR) :- wind_blowing_afternoon(C,DIR,SPEED), adjacent_to_considered(C),wind_speed_level(SPEED),wind_dir(DIR).

wind_blowing_morning_speed(C,SPEED) :- wind_blowing_morning(C,DIR,SPEED), location_considered(C),wind_speed_level(SPEED),wind_dir(DIR).
wind_blowing_afternoon_speed(C,SPEED) :- wind_blowing_afternoon(C,DIR,SPEED), location_considered(C),wind_speed_level(SPEED),wind_dir(DIR).

wind_blowing_morning_neighbour_speed(C,SPEED) :- wind_blowing_morning(C,DIR,SPEED), adjacent_to_considered(C),wind_speed_level(SPEED),wind_dir(DIR).
wind_blowing_afternoon_neighbour_speed(C,SPEED) :- wind_blowing_afternoon(C,DIR,SPEED), adjacent_to_considered(C),wind_speed_level(SPEED),wind_dir(DIR).


rain_lvs(0).
rain_lvs(1).
rain_lvs(2).
rain_lvs(4).
rain_lvs(6).

not_city_covered_less_than_(X,LV) :- not city_covered_less_than_(X,LV),lv(LV), location_considered(X).
not_city_covered_at_least_morning_(X,LV) :- not city_covered_at_least_morning_(X,LV),lv(LV), location_considered(X).
not_city_covered_at_least_afternoon_(X,LV) :- not city_covered_at_least_afternoon_(X,LV),lv(LV), location_considered(X).

not_city_covered_less_than_neighbour(X,LV) :- not city_covered_less_than_neighbour(X,LV),lv(LV), adjacent_to_considered(X).
not_city_covered_at_least_morning_neighbour(X,LV) :- not city_covered_at_least_morning_neighbour(X,LV),lv(LV), adjacent_to_considered(X).
not_city_covered_at_least_afternoon_neighbour(X,LV) :- not city_covered_at_least_afternoon_neighbour(X,LV),lv(LV), adjacent_to_considered(X).

% forecasted 0  forbid 1,2,4,6
:- rains_at(X,1,S), forecasted_rain(X,0,S), season(S), location_considered(X).
:- rains_at(X,2,S), forecasted_rain(X,0,S), season(S), location_considered(X).
:- rains_at(X,4,S), forecasted_rain(X,0,S), season(S), location_considered(X).
:- rains_at(X,6,S), forecasted_rain(X,0,S), season(S), location_considered(X).

% forecasted 1  forbid 0,2,4,6
:- rains_at(X,0,S), forecasted_rain(X,1,S), season(S), location_considered(X).
:- rains_at(X,2,S), forecasted_rain(X,1,S), season(S), location_considered(X).
:- rains_at(X,4,S), forecasted_rain(X,1,S), season(S), location_considered(X).
:- rains_at(X,6,S), forecasted_rain(X,1,S), season(S), location_considered(X).

% forecasted 2 forbid 0,1,4,6
:- rains_at(X,0,S), forecasted_rain(X,2,S), season(S), location_considered(X).
:- rains_at(X,1,S), forecasted_rain(X,2,S), season(S), location_considered(X).
:- rains_at(X,4,S), forecasted_rain(X,2,S), season(S), location_considered(X).
:- rains_at(X,6,S), forecasted_rain(X,2,S), season(S), location_considered(X).

% forecasted 4 forbid 0,1,2,6
:- rains_at(X,0,S), forecasted_rain(X,4,S), season(S), location_considered(X).
:- rains_at(X,1,S), forecasted_rain(X,4,S), season(S), location_considered(X).
:- rains_at(X,2,S), forecasted_rain(X,4,S), season(S), location_considered(X).
:- rains_at(X,6,S), forecasted_rain(X,4,S), season(S), location_considered(X).

% forecasted 6 forbid 0,1,2,4
:- rains_at(X,0,S), forecasted_rain(X,6,S), season(S), location_considered(X).
:- rains_at(X,1,S), forecasted_rain(X,6,S), season(S), location_considered(X).
:- rains_at(X,2,S), forecasted_rain(X,6,S), season(S), location_considered(X).
:- rains_at(X,4,S), forecasted_rain(X,6,S), season(S), location_considered(X).




rains_at(X,LV,S) :- forecasted_rain(X,LV,S), season(S), location_considered(X),rain_lvs(LV).

:- humidity_increased_at_afternoon(X), humidity_decreased_at_afternoon(X), adjacent_to_considered(X).
:- temperature_increased_at_afternoon(X),temperature_decreased_at_afternoon(X), adjacent_to_considered(X).

:- wind_blowing_morning_(X,D1),wind_blowing_morning_(X,D2), location_considered(X),wind_dir(D1), wind_dir(D2), D1!=D2.
:- wind_blowing_afternoon_(X,D1),wind_blowing_afternoon_(X,D2), location_considered(X),wind_dir(D1), wind_dir(D2), D1!=D2.


:- wind_blowing_morning_neighbour(X,D1),wind_blowing_morning_neighbour(X,D2,S2), adjacent_to_considered(X),wind_dir(D1), wind_dir(D2), D1!=D2.
:- wind_blowing_afternoon_neighbour(X,D1),wind_blowing_afternoon_neighbour(X,D2,S2), adjacent_to_considered(X),wind_dir(D1), wind_dir(D2), D1!=D2.


temp_front_afternoon_at_1000hPa(X,C) :- temp_front_afternoon_at_100m(X,C), location_considered(X), adjacent_to_considered(C).
temp_front_afternoon_at_850hPa(X,C) :- temp_front_afternoon_at_1_5km(X,C), location_considered(X), adjacent_to_considered(C).
temp_front_afternoon_at_500hPa(X,C) :- temp_front_afternoon_at_5_5km(X,C), location_considered(X), adjacent_to_considered(C).
temp_front_afternoon_at_925hPa(X,C) :- temp_front_afternoon_at_900m(X,C), location_considered(X), adjacent_to_considered(C).
temp_front_afternoon_at_700hPa(X,C) :- temp_front_afternoon_at_3km(X,C) , location_considered(X), adjacent_to_considered(C).
temp_front_afternoon_at_300hPa(X,C) :- temp_front_afternoon_at_9km(X,C), location_considered(X), adjacent_to_considered(C).
%lazy
hum_front_afternoon_at_1000hPa(X,C) :- hum_front_afternoon_at_100m(C,X), location_considered(X), adjacent_to_considered(C).
hum_front_afternoon_at_850hPa(X,C) :- hum_front_afternoon_at_1_5km(C,X), location_considered(X), adjacent_to_considered(C).
hum_front_afternoon_at_500hPa(X,C) :- hum_front_afternoon_at_5_5km(C,X), location_considered(X), adjacent_to_considered(C).
hum_front_afternoon_at_925hPa(X,C) :- hum_temp_front_afternoon_at_900m(C,X), location_considered(X), adjacent_to_considered(C).
hum_front_afternoon_at_700hPa(X,C) :- hum_temp_front_afternoon_at_3km(C,X) , location_considered(X), adjacent_to_considered(C).
hum_front_afternoon_at_300hPa(X,C) :- hum_temp_front_afternoon_at_9km(C,X), location_considered(X), adjacent_to_considered(C).

temp_front_morning_at_1000hPa(X,C) :- temp_front_morning_at_100m(X,C), location_considered(X), adjacent_to_considered(C).
temp_front_morning_at_850hPa(X,C) :- temp_front_morning_at_1_5km(X,C), location_considered(X), adjacent_to_considered(C).
temp_front_morning_at_500hPa(X,C) :- temp_front_morning_at_5_5km(X,C), location_considered(X), adjacent_to_considered(C).
temp_front_morning_at_925hPa(X,C) :- temp_front_morning_at_900m(X,C), location_considered(X), adjacent_to_considered(C).
temp_front_morning_at_700hPa(X,C) :- temp_front_morning_at_3km(X,C) , location_considered(X), adjacent_to_considered(C).
temp_front_morning_at_300hPa(X,C) :- temp_front_morning_at_9km(X,C), location_considered(X), adjacent_to_considered(C).
%lazy
hum_front_morning_at_1000hPa(X,C) :- hum_front_morning_at_100m(C,X), location_considered(X), adjacent_to_considered(C).
hum_front_morning_at_850hPa(X,C) :- hum_front_morning_at_1_5km(C,X), location_considered(X), adjacent_to_considered(C).
hum_front_morning_at_500hPa(X,C) :- hum_front_morning_at_5_5km(C,X), location_considered(X), adjacent_to_considered(C).
hum_front_morning_at_925hPa(X,C) :- hum_front_morning_at_900m(C,X), location_considered(X), adjacent_to_considered(C).
hum_front_morning_at_700hPa(X,C) :- hum_front_morning_at_3km(C,X) , location_considered(X), adjacent_to_considered(C).
hum_front_morning_at_300hPa(X,C) :- hum_front_morning_at_9km(C,X), location_considered(X), adjacent_to_considered(C).
