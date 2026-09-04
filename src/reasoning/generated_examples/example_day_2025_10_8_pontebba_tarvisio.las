% Example generated data for day (2025, 10, 8)

#pos(e80@1000,{ 

forecasted_sky(pontebba_tarvisio, "partly_cloudy", autumn),
forecasted_rain(pontebba_tarvisio, 0, autumn)},
{
sunny_at(pontebba_tarvisio,autumn), 
covered_at(pontebba_tarvisio,autumn), 
rains_at(pontebba_tarvisio,1,autumn), 
rains_at(pontebba_tarvisio,2,autumn), 
rains_at(pontebba_tarvisio,4,autumn), 
rains_at(pontebba_tarvisio,6,autumn)
},
{
location_considered(pontebba_tarvisio). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 10, 8).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_3km_covers(lignano_grado,1753, 21).
cloud_at_3km_covers(udine_palmanova,1753, 21).
cloud_at_5_5km_covers(barcis,1705, 9).
cloud_at_5_5km_covers(pordenone,1705, 9).
cloud_at_5_5km_covers(sappada_forni_villa,1707, 10).
cloud_at_5_5km_covers(barcis,1710, 12).
cloud_at_5_5km_covers(pordenone,1710, 12).
cloud_at_5_5km_covers(gemona_stolvizza,1705, 12).
cloud_at_5_5km_covers(pontebba_tarvisio,1711, 13).
cloud_at_5_5km_covers(pordenone,1712, 13).
cloud_at_5_5km_covers(sappada_forni_villa,1711, 14).
cloud_at_5_5km_covers(sappada_forni_villa,1711, 15).
cloud_at_5_5km_covers(gemona_stolvizza,1711, 15).
cloud_at_5_5km_covers(sappada_forni_villa,1711, 16).
cloud_at_5_5km_covers(pontebba_tarvisio,1711, 16).
cloud_at_5_5km_covers(udine_palmanova,1711, 16).
cloud_at_5_5km_covers(gorizia,1711, 16).
cloud_at_5_5km_covers(gemona_stolvizza,1711, 16).
cloud_at_5_5km_covers(lignano_grado,1711, 17).
cloud_at_5_5km_covers(udine_palmanova,1711, 17).
cloud_at_5_5km_covers(gorizia,1711, 17).
cloud_at_5_5km_covers(gemona_stolvizza,1711, 17).
cloud_at_5_5km_covers(pontebba_tarvisio,1719, 18).
cloud_at_5_5km_covers(gorizia,1719, 18).
cloud_at_5_5km_covers(trieste,1719, 18).
cloud_at_5_5km_covers(gemona_stolvizza,1719, 18).
cloud_at_5_5km_covers(trieste,1719, 19).
cloud_at_9km_covers(sappada_forni_villa,2011, 0).
cloud_at_9km_covers(sappada_forni_villa,2008, 1).
cloud_at_9km_covers(pontebba_tarvisio,2008, 1).
cloud_at_9km_covers(barcis,2008, 1).
cloud_at_9km_covers(udine_palmanova,2008, 1).
cloud_at_9km_covers(gemona_stolvizza,2008, 1).
cloud_at_9km_covers(pontebba_tarvisio,2008, 2).
cloud_at_9km_covers(lignano_grado,2008, 2).
cloud_at_9km_covers(barcis,2008, 2).
cloud_at_9km_covers(udine_palmanova,2008, 2).
cloud_at_9km_covers(gemona_stolvizza,2008, 2).
cloud_at_9km_covers(lignano_grado,2016, 3).
cloud_at_9km_covers(gorizia,2016, 3).
cloud_at_9km_covers(trieste,2016, 3).
cloud_at_9km_covers(trieste,2016, 4).
cloud_at_9km_covers(barcis,2014, 5).
cloud_at_9km_covers(trieste,2016, 7).
cloud_at_9km_covers(barcis,2021, 9).
cloud_at_9km_covers(pontebba_tarvisio,2023, 12).
cloud_at_9km_covers(pontebba_tarvisio,2028, 13).
cloud_at_9km_covers(gorizia,2030, 13).
cloud_at_9km_covers(trieste,2030, 13).
cloud_at_9km_covers(pontebba_tarvisio,2031, 14).
cloud_at_9km_covers(trieste,2031, 14).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,278.07).
% temperature_at_afternoon(sappada_forni_villa,278.60).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,277.80).
% temperature_at_afternoon(pontebba_tarvisio,278.12).
temperature_increased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,278.80).
% temperature_at_afternoon(lignano_grado,278.15).
temperature_decreased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,277.13).
% temperature_at_afternoon(barcis,278.25).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,278.97).
% temperature_at_afternoon(udine_palmanova,278.50).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,278.80).
% temperature_at_afternoon(gorizia,278.88).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,279.00).
% temperature_at_afternoon(trieste,278.92).
temperature_decreased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,278.73).
% temperature_at_afternoon(gemona_stolvizza,278.77).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,277.80).
% temperature_at_afternoon(pordenone,278.08).
temperature_increased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,51.33).
% humidity_at_afternoon(sappada_forni_villa,31.25).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,46.00).
% humidity_at_afternoon(pontebba_tarvisio,31.25).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,62.67).
% humidity_at_afternoon(lignano_grado,60.00).
humidity_decreased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,55.33).
% humidity_at_afternoon(barcis,45.42).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,76.00).
% humidity_at_afternoon(udine_palmanova,50.83).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,74.00).
% humidity_at_afternoon(gorizia,58.33).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,60.67).
% humidity_at_afternoon(trieste,55.00).
humidity_decreased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,70.00).
% humidity_at_afternoon(gemona_stolvizza,52.92).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,55.33).
% humidity_at_afternoon(pordenone,64.17).
humidity_increased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"S",21).
wind_blowing_afternoon(sappada_forni_villa,"SE",19).
wind_blowing_morning(pontebba_tarvisio,"S",23).
wind_blowing_afternoon(pontebba_tarvisio,"SE",20).
wind_blowing_morning(lignano_grado,"S",21).
wind_blowing_afternoon(lignano_grado,"SE",19).
wind_blowing_morning(barcis,"S",21).
wind_blowing_afternoon(barcis,"SE",19).
wind_blowing_morning(udine_palmanova,"S",23).
wind_blowing_afternoon(udine_palmanova,"SE",20).
wind_blowing_morning(gorizia,"S",23).
wind_blowing_afternoon(gorizia,"SE",20).
wind_blowing_morning(trieste,"S",23).
wind_blowing_afternoon(trieste,"SE",20).
wind_blowing_morning(gemona_stolvizza,"S",23).
wind_blowing_afternoon(gemona_stolvizza,"SE",20).
wind_blowing_morning(pordenone,"S",21).
wind_blowing_afternoon(pordenone,"SE",19).

temp_front_afternoon_at_100m(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_100m(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_750m(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_750m(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_1_4km(lignano_grado,trieste).
temp_front_afternoon_at_1_4km(gorizia,lignano_grado).
temp_front_afternoon_at_1_4km(gemona_stolvizza,gorizia).
temp_front_morning_at_3km(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_3km(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_3km(gemona_stolvizza,sappada_forni_villa).
temp_front_morning_at_3km(pordenone,udine_palmanova).
temp_front_afternoon_at_3km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_3km(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_3km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_5_5km(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_5_5km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_5_5km(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_5_5km(pordenone,sappada_forni_villa).

hum_front_morning_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_100m(barcis,pordenone).
hum_front_morning_at_100m(gorizia,pontebba_tarvisio).
hum_front_morning_at_100m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(gorizia,lignano_grado).
hum_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_morning_at_750m(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_750m(gemona_stolvizza,udine_palmanova).
hum_front_morning_at_750m(lignano_grado,udine_palmanova).
hum_front_morning_at_750m(gorizia,lignano_grado).
hum_front_morning_at_750m(pordenone,udine_palmanova).
hum_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_750m(lignano_grado,udine_palmanova).
hum_front_afternoon_at_750m(gorizia,lignano_grado).
hum_front_afternoon_at_750m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_750m(pordenone,udine_palmanova).
hum_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
hum_front_morning_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
hum_front_morning_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_1_4km(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_1_4km(barcis,pordenone).
hum_front_afternoon_at_1_4km(pordenone,sappada_forni_villa).
hum_front_afternoon_at_1_4km(lignano_grado,trieste).
hum_front_afternoon_at_1_4km(gorizia,lignano_grado).
hum_front_afternoon_at_1_4km(gemona_stolvizza,gorizia).

}). 
