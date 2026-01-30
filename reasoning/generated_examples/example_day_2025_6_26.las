% Example generated data for day (2025, 6, 26)

#pos(e24@100,{ 

% date(2025,6,26),

forecasted_sky(sappada_forni_villa, "partly_cloudy"),
forecasted_sky(pontebba_tarvisio, "mostly_clear"),
forecasted_sky(lignano_grado, "sunny"),
forecasted_sky(barcis, "partly_cloudy"),
forecasted_sky(udine_palmanova, "mostly_clear"),
forecasted_sky(gorizia, "mostly_clear"),
forecasted_sky(trieste, "sunny"),
forecasted_sky(gemona_stolvizza, "partly_cloudy"),
forecasted_sky(pordenone, "mostly_clear")
},
{
sunny_at(sappada_forni_villa), 
covered_at(sappada_forni_villa), 
partially_sunny_at(pontebba_tarvisio), 
covered_at(pontebba_tarvisio), 
partially_sunny_at(lignano_grado), 
covered_at(lignano_grado), 
sunny_at(barcis), 
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
date(2025, 6, 26).
 %to drive the season (winter, spring, summer, autumn)

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_1_4km_covers(pordenone,695, 4).
cloud_at_1_4km_covers(gorizia,696, 11).
cloud_at_1_4km_covers(sappada_forni_villa,703, 19).
cloud_at_1_4km_covers(pontebba_tarvisio,705, 21).
cloud_at_3km_covers(sappada_forni_villa,819, 22).
cloud_at_5_5km_covers(sappada_forni_villa,813, 16).
cloud_at_5_5km_covers(gemona_stolvizza,813, 16).
cloud_at_5_5km_covers(sappada_forni_villa,817, 19).
cloud_at_5_5km_covers(sappada_forni_villa,821, 20).
cloud_at_5_5km_covers(barcis,821, 20).
cloud_at_9km_covers(barcis,908, 10).
cloud_at_9km_covers(gemona_stolvizza,908, 11).
cloud_at_9km_covers(sappada_forni_villa,919, 12).
cloud_at_9km_covers(barcis,919, 12).
cloud_at_9km_covers(pordenone,923, 13).
cloud_at_9km_covers(barcis,937, 15).
cloud_at_9km_covers(sappada_forni_villa,937, 16).
cloud_at_9km_covers(gemona_stolvizza,937, 16).
cloud_at_9km_covers(barcis,932, 16).
cloud_at_9km_covers(sappada_forni_villa,942, 19).
cloud_at_9km_covers(pontebba_tarvisio,942, 19).
cloud_at_9km_covers(gemona_stolvizza,942, 19).
cloud_at_9km_covers(sappada_forni_villa,942, 20).
cloud_at_9km_covers(barcis,942, 20).
cloud_at_9km_covers(pordenone,942, 20).
cloud_at_9km_covers(lignano_grado,942, 21).
cloud_at_9km_covers(barcis,942, 21).
cloud_at_9km_covers(udine_palmanova,942, 21).
cloud_at_9km_covers(sappada_forni_villa,942, 22).
cloud_at_9km_covers(pontebba_tarvisio,942, 22).
cloud_at_9km_covers(lignano_grado,942, 22).
cloud_at_9km_covers(barcis,942, 22).
cloud_at_9km_covers(udine_palmanova,942, 22).
cloud_at_9km_covers(gorizia,942, 22).
cloud_at_9km_covers(gemona_stolvizza,942, 22).
cloud_at_9km_covers(sappada_forni_villa,942, 23).
cloud_at_9km_covers(pontebba_tarvisio,942, 23).
cloud_at_9km_covers(lignano_grado,942, 23).
cloud_at_9km_covers(barcis,942, 23).
cloud_at_9km_covers(udine_palmanova,942, 23).
cloud_at_9km_covers(trieste,942, 23).
cloud_at_9km_covers(gemona_stolvizza,942, 23).
cloud_at_9km_covers(pordenone,942, 23).

%summing up temperature and humidity facts 

temperature_at_morning(sappada_forni_villa,277.07).
temperature_at_afternoon(sappada_forni_villa,277.52).
temperature_at_morning(pontebba_tarvisio,275.17).
temperature_at_afternoon(pontebba_tarvisio,277.35).
temperature_at_morning(lignano_grado,277.70).
temperature_at_afternoon(lignano_grado,276.75).
temperature_at_morning(barcis,277.83).
temperature_at_afternoon(barcis,277.56).
temperature_at_morning(udine_palmanova,276.87).
temperature_at_afternoon(udine_palmanova,275.75).
temperature_at_morning(gorizia,277.80).
temperature_at_afternoon(gorizia,276.19).
temperature_at_morning(trieste,278.13).
temperature_at_afternoon(trieste,276.81).
temperature_at_morning(gemona_stolvizza,278.27).
temperature_at_afternoon(gemona_stolvizza,277.21).
temperature_at_morning(pordenone,277.03).
temperature_at_afternoon(pordenone,276.10).
humidity_at_morning(sappada_forni_villa,35.33).
humidity_at_afternoon(sappada_forni_villa,42.50).
humidity_at_morning(pontebba_tarvisio,41.33).
humidity_at_afternoon(pontebba_tarvisio,51.25).
humidity_at_morning(lignano_grado,44.67).
humidity_at_afternoon(lignano_grado,45.42).
humidity_at_morning(barcis,41.33).
humidity_at_afternoon(barcis,44.58).
humidity_at_morning(udine_palmanova,59.33).
humidity_at_afternoon(udine_palmanova,42.92).
humidity_at_morning(gorizia,52.00).
humidity_at_afternoon(gorizia,45.83).
humidity_at_morning(trieste,60.67).
humidity_at_afternoon(trieste,48.33).
humidity_at_morning(gemona_stolvizza,44.00).
humidity_at_afternoon(gemona_stolvizza,40.00).
humidity_at_morning(pordenone,48.67).
humidity_at_afternoon(pordenone,61.25).

% Humidity front data:
% humidty_front(location_1,location_2,hh): between the two locations there's a sharp change 
}). 
