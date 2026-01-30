% Example generated data for day (2025, 6, 25)

#pos(e23@100,{ 

% date(2025,6,25),

forecasted_sky(sappada_forni_villa, "mostly_clear"),
forecasted_sky(pontebba_tarvisio, "mostly_clear"),
forecasted_sky(lignano_grado, "sunny"),
forecasted_sky(barcis, "mostly_clear"),
forecasted_sky(udine_palmanova, "sunny"),
forecasted_sky(gorizia, "sunny"),
forecasted_sky(trieste, "sunny"),
forecasted_sky(gemona_stolvizza, "mostly_clear"),
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
partially_sunny_at(udine_palmanova), 
covered_at(udine_palmanova), 
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
date(2025, 6, 25).
 %to drive the season (winter, spring, summer, autumn)

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_1_4km_covers(sappada_forni_villa,694, 21).
cloud_at_3km_covers(gemona_stolvizza,801, 12).
cloud_at_5_5km_covers(barcis,794, 8).
cloud_at_9km_covers(lignano_grado,878, 1).
cloud_at_9km_covers(gorizia,878, 1).
cloud_at_9km_covers(trieste,878, 1).
cloud_at_9km_covers(trieste,878, 2).
cloud_at_9km_covers(barcis,883, 10).
cloud_at_9km_covers(barcis,883, 11).
cloud_at_9km_covers(gemona_stolvizza,887, 12).
cloud_at_9km_covers(barcis,886, 13).
cloud_at_9km_covers(gemona_stolvizza,887, 13).
cloud_at_9km_covers(gemona_stolvizza,887, 14).

%summing up temperature and humidity facts 

temperature_at_morning(sappada_forni_villa,275.07).
temperature_at_afternoon(sappada_forni_villa,278.48).
temperature_at_morning(pontebba_tarvisio,273.00).
temperature_at_afternoon(pontebba_tarvisio,278.98).
temperature_at_morning(lignano_grado,278.60).
temperature_at_afternoon(lignano_grado,278.56).
temperature_at_morning(barcis,277.77).
temperature_at_afternoon(barcis,278.69).
temperature_at_morning(udine_palmanova,276.73).
temperature_at_afternoon(udine_palmanova,277.42).
temperature_at_morning(gorizia,277.90).
temperature_at_afternoon(gorizia,278.48).
temperature_at_morning(trieste,278.70).
temperature_at_afternoon(trieste,278.67).
temperature_at_morning(gemona_stolvizza,278.33).
temperature_at_afternoon(gemona_stolvizza,278.75).
temperature_at_morning(pordenone,277.00).
temperature_at_afternoon(pordenone,277.38).
humidity_at_morning(sappada_forni_villa,52.67).
humidity_at_afternoon(sappada_forni_villa,39.58).
humidity_at_morning(pontebba_tarvisio,56.67).
humidity_at_afternoon(pontebba_tarvisio,37.08).
humidity_at_morning(lignano_grado,52.00).
humidity_at_afternoon(lignano_grado,55.42).
humidity_at_morning(barcis,45.33).
humidity_at_afternoon(barcis,35.42).
humidity_at_morning(udine_palmanova,52.00).
humidity_at_afternoon(udine_palmanova,48.75).
humidity_at_morning(gorizia,56.67).
humidity_at_afternoon(gorizia,53.33).
humidity_at_morning(trieste,48.00).
humidity_at_afternoon(trieste,57.08).
humidity_at_morning(gemona_stolvizza,44.00).
humidity_at_afternoon(gemona_stolvizza,38.33).
humidity_at_morning(pordenone,54.00).
humidity_at_afternoon(pordenone,47.08).

% Humidity front data:
% humidty_front(location_1,location_2,hh): between the two locations there's a sharp change 
}). 
