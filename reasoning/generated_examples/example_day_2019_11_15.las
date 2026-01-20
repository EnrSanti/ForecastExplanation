% Example generated data for day (2019, 11, 15)

#pos(e24,{ 

% date(2019,11,15),

forecasted_sky(sappada_forni_villa, "cloudy"),
forecasted_sky(pontebba_tarvisio, "cloudy"),
forecasted_sky(lignano_grado, "cloudy"),
forecasted_sky(barcis, "cloudy"),
forecasted_sky(udine_palamnova, "cloudy"),
forecasted_sky(gorizia, "cloudy"),
forecasted_sky(trieste, "mostly_cloudy"),
forecasted_sky(gemona_stolvizza, "cloudy"),
forecasted_sky(pordenone, "cloudy")
},
{
sunny_at(sappada_forni_villa), 
partially_sunny_at(sappada_forni_villa), 
sunny_at(pontebba_tarvisio), 
partially_sunny_at(pontebba_tarvisio), 
sunny_at(lignano_grado), 
partially_sunny_at(lignano_grado), 
sunny_at(barcis), 
partially_sunny_at(barcis), 
sunny_at(udine_palamnova), 
partially_sunny_at(udine_palamnova), 
sunny_at(gorizia), 
partially_sunny_at(gorizia), 
sunny_at(trieste), 
partially_sunny_at(trieste), 
sunny_at(gemona_stolvizza), 
partially_sunny_at(gemona_stolvizza), 
sunny_at(pordenone), 
partially_sunny_at(pordenone)
},
{
date(2019, 11, 15).
 %to drive the season (winter, spring, summer, autumn)

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)

%summing up temperature and humidity facts 

wind_blowing_morning(sappada_forni_villa,N,34.033).
wind_blowing_afternoon(sappada_forni_villa,N,34.282).
wind_blowing_morning(pontebba_tarvisio,N,33.367).
wind_blowing_afternoon(pontebba_tarvisio,N,34.682).
wind_blowing_morning(lignano_grado,N,31.651).
wind_blowing_afternoon(lignano_grado,N,36.945).
wind_blowing_morning(barcis,N,35.769).
wind_blowing_afternoon(barcis,N,35.409).
wind_blowing_morning(udine_palamnova,N,33.468).
wind_blowing_afternoon(udine_palamnova,N,36.867).
wind_blowing_morning(gorizia,N,33.468).
wind_blowing_afternoon(gorizia,N,36.867).
wind_blowing_morning(trieste,NE,30.070).
wind_blowing_afternoon(trieste,NE,37.210).
wind_blowing_morning(gemona_stolvizza,N,33.468).
wind_blowing_afternoon(gemona_stolvizza,N,36.867).
wind_blowing_morning(pordenone,N,35.769).
wind_blowing_afternoon(pordenone,N,35.409).

% Humidity front data:
% humidty_front(location_1,location_2,hh): between the two locations there's a sharp change 
}). 
