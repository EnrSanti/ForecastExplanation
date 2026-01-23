% Example generated data for day (2019, 4, 1)

#pos(e15@100,{ 

% date(2019,4,1),

forecasted_sky(sappada_forni_villa, "partly_cloudy"),
forecasted_sky(pontebba_tarvisio, "partly_cloudy"),
forecasted_sky(lignano_grado, "sunny"),
forecasted_sky(barcis, "mostly_clear"),
forecasted_sky(udine_palmanova, "mostly_clear"),
forecasted_sky(gorizia, "sunny"),
forecasted_sky(trieste, "sunny"),
forecasted_sky(gemona_stolvizza, "partly_cloudy"),
forecasted_sky(pordenone, "sunny")
},
{
sunny_at(sappada_forni_villa), 
covered_at(sappada_forni_villa), 
sunny_at(pontebba_tarvisio), 
covered_at(pontebba_tarvisio), 
partially_sunny_at(lignano_grado), 
covered_at(lignano_grado), 
partially_sunny_at(barcis), 
covered_at(barcis), 
partially_sunny_at(udine_palmanova), 
covered_at(udine_palmanova), 
partially_sunny_at(gorizia), 
covered_at(gorizia), 
partially_sunny_at(trieste), 
covered_at(trieste), 
sunny_at(gemona_stolvizza), 
covered_at(gemona_stolvizza), 
partially_sunny_at(pordenone), 
covered_at(pordenone)
},
{
date(2019, 4, 1).
 %to drive the season (winter, spring, summer, autumn)

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_9km_covers(barcis,7, 23).

%summing up temperature and humidity facts 

temperature_at_morning(sappada_forni_villa,264.97).
temperature_at_afternoon(sappada_forni_villa,265.71).
temperature_at_morning(pontebba_tarvisio,267.23).
temperature_at_afternoon(pontebba_tarvisio,267.27).
temperature_at_morning(lignano_grado,262.30).
temperature_at_afternoon(lignano_grado,262.19).
temperature_at_morning(barcis,263.90).
temperature_at_afternoon(barcis,264.71).
temperature_at_morning(udine_palmanova,262.40).
temperature_at_afternoon(udine_palmanova,263.21).
temperature_at_morning(gorizia,262.40).
temperature_at_afternoon(gorizia,263.54).
temperature_at_morning(trieste,262.00).
temperature_at_afternoon(trieste,262.83).
temperature_at_morning(gemona_stolvizza,264.00).
temperature_at_afternoon(gemona_stolvizza,265.77).
temperature_at_morning(pordenone,262.67).
temperature_at_afternoon(pordenone,262.85).
humidity_at_morning(sappada_forni_villa,57.33).
humidity_at_afternoon(sappada_forni_villa,51.25).
humidity_at_morning(pontebba_tarvisio,39.33).
humidity_at_afternoon(pontebba_tarvisio,42.92).
humidity_at_morning(lignano_grado,82.00).
humidity_at_afternoon(lignano_grado,70.42).
humidity_at_morning(barcis,68.67).
humidity_at_afternoon(barcis,79.58).
humidity_at_morning(udine_palmanova,78.00).
humidity_at_afternoon(udine_palmanova,80.42).
humidity_at_morning(gorizia,63.33).
humidity_at_afternoon(gorizia,85.00).
humidity_at_morning(trieste,60.00).
humidity_at_afternoon(trieste,80.00).
humidity_at_morning(gemona_stolvizza,78.00).
humidity_at_afternoon(gemona_stolvizza,73.33).
humidity_at_morning(pordenone,71.33).
humidity_at_afternoon(pordenone,55.42).
wind_blowing_morning(sappada_forni_villa,E,8.886).
wind_blowing_afternoon(sappada_forni_villa,E,19.479).
wind_blowing_morning(pontebba_tarvisio,E,7.076).
wind_blowing_afternoon(pontebba_tarvisio,E,16.845).
wind_blowing_morning(lignano_grado,E,13.435).
wind_blowing_afternoon(lignano_grado,E,22.224).
wind_blowing_morning(barcis,E,11.462).
wind_blowing_afternoon(barcis,E,21.241).
wind_blowing_morning(udine_palmanova,E,9.163).
wind_blowing_afternoon(udine_palmanova,E,19.632).
wind_blowing_morning(gorizia,E,9.163).
wind_blowing_afternoon(gorizia,E,19.632).
wind_blowing_morning(trieste,E,11.797).
wind_blowing_afternoon(trieste,E,21.691).
wind_blowing_morning(gemona_stolvizza,E,9.163).
wind_blowing_afternoon(gemona_stolvizza,E,19.632).
wind_blowing_morning(pordenone,E,11.462).
wind_blowing_afternoon(pordenone,E,21.241).

% Humidity front data:
% humidty_front(location_1,location_2,hh): between the two locations there's a sharp change 
}). 
