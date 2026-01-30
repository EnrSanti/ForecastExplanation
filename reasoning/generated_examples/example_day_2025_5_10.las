% Example generated data for day (2025, 5, 10)

#pos(e17@100,{ 

% date(2025,5,10),

forecasted_sky(sappada_forni_villa, "partly_cloudy"),
forecasted_sky(pontebba_tarvisio, "partly_cloudy"),
forecasted_sky(lignano_grado, "mostly_clear"),
forecasted_sky(barcis, "partly_cloudy"),
forecasted_sky(udine_palmanova, "mostly_clear"),
forecasted_sky(gorizia, "mostly_clear"),
forecasted_sky(trieste, "mostly_clear"),
forecasted_sky(gemona_stolvizza, "partly_cloudy"),
forecasted_sky(pordenone, "mostly_clear")
},
{
sunny_at(sappada_forni_villa), 
covered_at(sappada_forni_villa), 
sunny_at(pontebba_tarvisio), 
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
date(2025, 5, 10).
 %to drive the season (winter, spring, summer, autumn)

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_100m_covers(pontebba_tarvisio,532, 1).
cloud_at_100m_covers(pontebba_tarvisio,532, 2).
cloud_at_750m_covers(pontebba_tarvisio,517, 1).
cloud_at_750m_covers(pontebba_tarvisio,517, 2).
cloud_at_1_4km_covers(pontebba_tarvisio,680, 0).
cloud_at_1_4km_covers(pontebba_tarvisio,680, 1).
cloud_at_1_4km_covers(pontebba_tarvisio,680, 2).
cloud_at_1_4km_covers(pontebba_tarvisio,680, 3).
cloud_at_3km_covers(pontebba_tarvisio,777, 5).
cloud_at_3km_covers(pontebba_tarvisio,777, 6).
cloud_at_3km_covers(gemona_stolvizza,777, 6).
cloud_at_5_5km_covers(sappada_forni_villa,773, 0).
cloud_at_5_5km_covers(pontebba_tarvisio,773, 0).
cloud_at_5_5km_covers(gemona_stolvizza,773, 0).
cloud_at_5_5km_covers(sappada_forni_villa,773, 1).
cloud_at_5_5km_covers(pontebba_tarvisio,773, 1).
cloud_at_5_5km_covers(barcis,773, 1).
cloud_at_5_5km_covers(gemona_stolvizza,773, 1).
cloud_at_5_5km_covers(barcis,784, 2).
cloud_at_5_5km_covers(gemona_stolvizza,783, 2).
cloud_at_5_5km_covers(gemona_stolvizza,783, 3).
cloud_at_5_5km_covers(udine_palmanova,783, 4).
cloud_at_5_5km_covers(gorizia,783, 4).
cloud_at_5_5km_covers(gemona_stolvizza,783, 4).
cloud_at_9km_covers(trieste,856, 0).
cloud_at_9km_covers(trieste,856, 2).

%summing up temperature and humidity facts 

temperature_at_morning(sappada_forni_villa,278.00).
temperature_at_afternoon(sappada_forni_villa,278.69).
temperature_at_morning(pontebba_tarvisio,278.73).
temperature_at_afternoon(pontebba_tarvisio,277.38).
temperature_at_morning(lignano_grado,276.73).
temperature_at_afternoon(lignano_grado,276.02).
temperature_at_morning(barcis,278.33).
temperature_at_afternoon(barcis,278.35).
temperature_at_morning(udine_palmanova,276.07).
temperature_at_afternoon(udine_palmanova,275.15).
temperature_at_morning(gorizia,275.17).
temperature_at_afternoon(gorizia,274.38).
temperature_at_morning(trieste,275.17).
temperature_at_afternoon(trieste,277.69).
temperature_at_morning(gemona_stolvizza,278.20).
temperature_at_afternoon(gemona_stolvizza,276.42).
temperature_at_morning(pordenone,277.50).
temperature_at_afternoon(pordenone,276.33).
humidity_at_morning(sappada_forni_villa,34.67).
humidity_at_afternoon(sappada_forni_villa,27.50).
humidity_at_morning(pontebba_tarvisio,28.67).
humidity_at_afternoon(pontebba_tarvisio,39.58).
humidity_at_morning(lignano_grado,48.00).
humidity_at_afternoon(lignano_grado,45.42).
humidity_at_morning(barcis,34.67).
humidity_at_afternoon(barcis,33.33).
humidity_at_morning(udine_palmanova,32.00).
humidity_at_afternoon(udine_palmanova,46.25).
humidity_at_morning(gorizia,38.67).
humidity_at_afternoon(gorizia,42.92).
humidity_at_morning(trieste,47.33).
humidity_at_afternoon(trieste,52.50).
humidity_at_morning(gemona_stolvizza,54.00).
humidity_at_afternoon(gemona_stolvizza,60.42).
humidity_at_morning(pordenone,47.33).
humidity_at_afternoon(pordenone,65.83).

% Humidity front data:
% humidty_front(location_1,location_2,hh): between the two locations there's a sharp change 
}). 
