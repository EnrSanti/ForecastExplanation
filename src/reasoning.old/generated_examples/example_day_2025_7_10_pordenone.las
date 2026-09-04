% Example generated data for day (2025, 7, 10)

#pos(e608@1000,{ 

forecasted_sky(pordenone, "sunny", summer),
forecasted_rain(pordenone, 0, summer)},
{
partially_sunny_at(pordenone,summer), 
covered_at(pordenone,summer), 
rains_at(pordenone,1,summer), 
rains_at(pordenone,2,summer), 
rains_at(pordenone,4,summer), 
rains_at(pordenone,6,summer)
},
{
location_considered(pordenone). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 7, 10).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_3km_covers(gemona_stolvizza,1295, 11).
cloud_at_3km_covers(barcis,1307, 14).
cloud_at_3km_covers(sappada_forni_villa,1307, 15).
cloud_at_3km_covers(barcis,1307, 15).
cloud_at_3km_covers(pontebba_tarvisio,1315, 17).
cloud_at_3km_covers(pordenone,1315, 18).
cloud_at_3km_covers(pordenone,1307, 19).
cloud_at_9km_covers(lignano_grado,1487, 9).
cloud_at_9km_covers(udine_palmanova,1487, 9).
cloud_at_9km_covers(gorizia,1487, 9).
cloud_at_9km_covers(trieste,1487, 9).
cloud_at_9km_covers(gemona_stolvizza,1487, 9).
cloud_at_9km_covers(trieste,1487, 10).
cloud_at_9km_covers(barcis,1492, 19).
cloud_at_9km_covers(pordenone,1492, 19).
cloud_at_9km_covers(lignano_grado,1492, 20).
cloud_at_9km_covers(barcis,1492, 20).
cloud_at_9km_covers(pordenone,1492, 20).
cloud_at_9km_covers(lignano_grado,1492, 21).
cloud_at_9km_covers(barcis,1492, 21).
cloud_at_9km_covers(udine_palmanova,1492, 21).
cloud_at_9km_covers(pordenone,1492, 21).
cloud_at_9km_covers(lignano_grado,1492, 22).
cloud_at_9km_covers(udine_palmanova,1492, 22).
cloud_at_9km_covers(barcis,1494, 22).
cloud_at_9km_covers(lignano_grado,1498, 23).
cloud_at_9km_covers(barcis,1498, 23).
cloud_at_9km_covers(udine_palmanova,1498, 23).
cloud_at_9km_covers(pordenone,1498, 23).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,274.10).
% temperature_at_afternoon(sappada_forni_villa,276.71).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,276.97).
% temperature_at_afternoon(pontebba_tarvisio,277.33).
temperature_increased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,274.40).
% temperature_at_afternoon(lignano_grado,274.25).
temperature_decreased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,275.00).
% temperature_at_afternoon(barcis,276.60).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,274.10).
% temperature_at_afternoon(udine_palmanova,274.52).
temperature_increased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,274.30).
% temperature_at_afternoon(gorizia,275.98).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,273.83).
% temperature_at_afternoon(trieste,276.44).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,274.87).
% temperature_at_afternoon(gemona_stolvizza,276.17).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,273.83).
% temperature_at_afternoon(pordenone,273.98).
temperature_increased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,64.00).
% humidity_at_afternoon(sappada_forni_villa,45.00).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,54.67).
% humidity_at_afternoon(pontebba_tarvisio,58.75).
humidity_increased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,62.00).
% humidity_at_afternoon(lignano_grado,51.25).
humidity_decreased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,59.33).
% humidity_at_afternoon(barcis,35.83).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,46.00).
% humidity_at_afternoon(udine_palmanova,49.17).
humidity_increased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,55.33).
% humidity_at_afternoon(gorizia,67.50).
humidity_increased_at_afternoon(gorizia).
% humidity_at_morning(trieste,43.33).
% humidity_at_afternoon(trieste,64.58).
humidity_increased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,64.00).
% humidity_at_afternoon(gemona_stolvizza,64.17).
humidity_increased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,52.00).
% humidity_at_afternoon(pordenone,36.25).
humidity_decreased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"SE",20).
wind_blowing_afternoon(sappada_forni_villa,"SE",24).
wind_blowing_morning(pontebba_tarvisio,"SE",20).
wind_blowing_afternoon(pontebba_tarvisio,"SE",27).
wind_blowing_morning(lignano_grado,"SE",19).
wind_blowing_afternoon(lignano_grado,"SE",23).
wind_blowing_morning(barcis,"SE",20).
wind_blowing_afternoon(barcis,"SE",24).
wind_blowing_morning(udine_palmanova,"SE",19).
wind_blowing_afternoon(udine_palmanova,"SE",24).
wind_blowing_morning(gorizia,"SE",19).
wind_blowing_afternoon(gorizia,"SE",24).
wind_blowing_morning(trieste,"SE",19).
wind_blowing_afternoon(trieste,"SE",24).
wind_blowing_morning(gemona_stolvizza,"SE",20).
wind_blowing_afternoon(gemona_stolvizza,"SE",27).
wind_blowing_morning(pordenone,"SE",19).
wind_blowing_afternoon(pordenone,"SE",23).

temp_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_100m(gorizia,pontebba_tarvisio).
temp_front_morning_at_100m(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_100m(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_3km(barcis,sappada_forni_villa).
temp_front_afternoon_at_3km(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_3km(pordenone,sappada_forni_villa).
temp_front_morning_at_5_5km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_5_5km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_5_5km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_9km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_9km(gemona_stolvizza,pontebba_tarvisio).

hum_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_100m(barcis,sappada_forni_villa).
hum_front_morning_at_100m(gorizia,pontebba_tarvisio).
hum_front_morning_at_100m(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_morning_at_750m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_750m(barcis,sappada_forni_villa).
hum_front_morning_at_750m(gorizia,pontebba_tarvisio).
hum_front_morning_at_750m(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_750m(barcis,pordenone).
hum_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_750m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_750m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_750m(pontebba_tarvisio,sappada_forni_villa).
hum_front_afternoon_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
hum_front_afternoon_at_1_4km(barcis,pordenone).
hum_front_afternoon_at_1_4km(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_1_4km(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_1_4km(pordenone,sappada_forni_villa).

}). 
