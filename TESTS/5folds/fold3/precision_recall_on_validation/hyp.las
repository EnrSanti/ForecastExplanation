

forecasted_rain(pordenone,1,spring) :- not_city_covered_at_least_morning_(pordenone,5), hum_front_morning_at_700hPa(pordenone,barcis).
forecasted_rain(trieste,2,spring) :- city_covered_at_least_morning_(trieste,1), not_city_covered_at_least_afternoon_(trieste,1).
forecasted_sky(udine_palmanova,"partly_cloudy",summer) :- wind_blowing_morning_(udine_palmanova,"E"), not_city_covered_at_least_afternoon_(udine_palmanova,3).
forecasted_sky(gemona_stolvizza,"partly_cloudy",autumn) :- humidity_increased_at_afternoon_neighbour(sappada_forni_villa), not_city_covered_at_least_morning_(gemona_stolvizza,1).
forecasted_sky(trieste,"mostly_clear",winter) :- city_covered_less_than_(trieste,1).
forecasted_rain(pontebba_tarvisio,4,spring) :- city_covered_at_least_morning_(pontebba_tarvisio,5), city_covered_exactly_morning_(pontebba_tarvisio,5).
forecasted_rain(barcis,2,spring) :- city_covered_at_least_afternoon_(barcis,4), wind_blowing_morning_(barcis,"NE").
forecasted_sky(pordenone,"partly_cloudy",autumn) :- wind_blowing_afternoon_(pordenone,"SE"), not_city_covered_at_least_morning_(pordenone,1).
forecasted_sky(gorizia,"cloudy",spring) :- city_covered_at_least_morning_(gorizia,5).
forecasted_rain(gorizia,1,autumn) :- wind_blowing_afternoon_(gorizia,"SE"), city_covered_less_than_(gorizia,3).
forecasted_rain(trieste,1,spring) :- temperature_increased_at_afternoon_(trieste), humidity_decreased_at_afternoon_(trieste), not_city_covered_at_least_afternoon_(trieste,2).
forecasted_sky(trieste,"mostly_cloudy",autumn) :- wind_blowing_afternoon_(trieste,"N"), city_covered_less_than_(trieste,5).
forecasted_rain(lignano_grado,4,autumn) :- city_covered_at_least_afternoon_(lignano_grado,1), temperature_increased_at_afternoon_(lignano_grado), temp_front_morning_at_1000hPa(lignano_grado,trieste).
forecasted_rain(pordenone,4,spring) :- not_city_covered_less_than_(pordenone,6), temp_front_morning_at_300hPa(pordenone,udine_palmanova).
forecasted_rain(lignano_grado,1,autumn) :- not_city_covered_less_than_(lignano_grado,6), city_covered_exactly_afternoon_(lignano_grado,3).
forecasted_rain(pordenone,6,winter) :- city_covered_at_least_morning_(pordenone,4), not_city_covered_at_least_afternoon_(pordenone,1).
forecasted_rain(trieste,1,winter) :- humidity_decreased_at_afternoon_neighbour(lignano_grado), not_city_covered_at_least_afternoon_(trieste,2).
forecasted_rain(lignano_grado,2,autumn) :- temperature_decreased_at_afternoon_neighbour(trieste), city_covered_at_least_morning_(lignano_grado,1), city_covered_less_than_(lignano_grado,6).
forecasted_sky(sappada_forni_villa,"mostly_cloudy",winter) :- city_covered_at_least_afternoon_(sappada_forni_villa,4).
forecasted_rain(sappada_forni_villa,1,spring) :- not_city_covered_at_least_afternoon_(sappada_forni_villa,4), hum_front_morning_at_1000hPa(sappada_forni_villa,pordenone).
forecasted_sky(gemona_stolvizza,"sunny",summer) :- not_city_covered_at_least_morning_(gemona_stolvizza,1), temp_front_morning_at_1000hPa(gemona_stolvizza,udine_palmanova).
forecasted_sky(udine_palmanova,"mostly_cloudy",winter) :- not_city_covered_at_least_morning_(udine_palmanova,1).
forecasted_rain(gemona_stolvizza,0,autumn) :- city_covered_at_least_afternoon_(gemona_stolvizza,3), wind_blowing_afternoon_(gemona_stolvizza,"SE").
forecasted_sky(pontebba_tarvisio,"cloudy",autumn) :- city_covered_at_least_morning_(pontebba_tarvisio,1), wind_blowing_afternoon_(pontebba_tarvisio,"N").
forecasted_sky(sappada_forni_villa,"sunny",autumn) :- not_city_covered_at_least_afternoon_(sappada_forni_villa,2), wind_blowing_afternoon_(sappada_forni_villa,"SE").
forecasted_sky(pordenone,"mostly_cloudy",autumn) :- city_covered_at_least_morning_(pordenone,1), temp_front_afternoon_at_1000hPa(pordenone,udine_palmanova).
forecasted_rain(gorizia,0,spring) :- city_covered_less_than_(gorizia,1).
forecasted_sky(pontebba_tarvisio,"partly_cloudy",spring) :- not_city_covered_at_least_morning_(pontebba_tarvisio,2).
forecasted_sky(pordenone,"mostly_cloudy",spring) :- not_city_covered_less_than_(pordenone,6).
forecasted_sky(pontebba_tarvisio,"partly_cloudy",summer) :- city_covered_at_least_morning_(pontebba_tarvisio,3), temperature_decreased_at_afternoon_(pontebba_tarvisio).
forecasted_rain(barcis,6,winter) :- not_city_covered_at_least_afternoon_(barcis,2), city_covered_at_least_morning_(barcis,5).
forecasted_sky(lignano_grado,"partly_cloudy",summer) :- not_city_covered_less_than_(lignano_grado,3).
forecasted_rain(udine_palmanova,4,summer) :- not_city_covered_less_than_(udine_palmanova,4).
forecasted_rain(pontebba_tarvisio,1,winter) :- city_covered_at_least_morning_(pontebba_tarvisio,1), wind_blowing_morning_(pontebba_tarvisio,"E").
forecasted_rain(gemona_stolvizza,0,autumn) :- not_city_covered_at_least_morning_(gemona_stolvizza,3), humidity_increased_at_afternoon_(gemona_stolvizza).
forecasted_sky(barcis,"mostly_clear",autumn) :- temperature_increased_at_afternoon_(barcis), not_city_covered_at_least_morning_(barcis,1).
forecasted_rain(lignano_grado,0,autumn) :- city_covered_less_than_(lignano_grado,2).
forecasted_sky(barcis,"partly_cloudy",summer) :- not_city_covered_less_than_(barcis,5), humidity_increased_at_afternoon_(barcis).
forecasted_sky(pontebba_tarvisio,"cloudy",autumn) :- city_covered_at_least_morning_(pontebba_tarvisio,6).
forecasted_rain(trieste,4,summer) :- not_city_covered_less_than_(trieste,1), city_covered_exactly_morning_(trieste,1).
forecasted_sky(pordenone,"mostly_cloudy",winter) :- city_covered_at_least_afternoon_(pordenone,1), temp_front_afternoon_at_1000hPa(pordenone,udine_palmanova).
forecasted_rain(udine_palmanova,6,autumn) :- city_covered_at_least_morning_(udine_palmanova,6).
forecasted_rain(barcis,6,autumn) :- wind_blowing_afternoon_(barcis,"N"), not_city_covered_at_least_afternoon_(barcis,2).
forecasted_sky(udine_palmanova,"mostly_cloudy",winter) :- city_covered_at_least_morning_(udine_palmanova,5).
forecasted_rain(gorizia,6,spring) :- city_covered_at_least_afternoon_(gorizia,5), city_covered_exactly_afternoon_(gorizia,5).
forecasted_rain(trieste,0,winter) :- city_covered_exactly_morning_(trieste,2), not_city_covered_less_than_(trieste,4).
forecasted_rain(pordenone,4,autumn) :- city_covered_at_least_afternoon_(pordenone,1), wind_blowing_afternoon_(pordenone,"N").
forecasted_rain(lignano_grado,1,winter) :- city_covered_at_least_morning_(lignano_grado,5), temp_front_morning_at_700hPa(lignano_grado,trieste).
forecasted_sky(udine_palmanova,"mostly_cloudy",autumn) :- not_city_covered_less_than_(udine_palmanova,4), hum_front_afternoon_at_1000hPa(udine_palmanova,lignano_grado).
forecasted_sky(lignano_grado,"sunny",autumn) :- not_city_covered_less_than_(lignano_grado,3), city_covered_less_than_(lignano_grado,4).
forecasted_sky(lignano_grado,"mostly_cloudy",autumn) :- not_city_covered_less_than_(lignano_grado,2), wind_blowing_morning_(lignano_grado,"E").
forecasted_rain(barcis,0,winter) :- city_covered_at_least_morning_(barcis,4), not_city_covered_at_least_morning_(barcis,5).
forecasted_rain(gemona_stolvizza,6,autumn) :- wind_blowing_afternoon_(gemona_stolvizza,"N"), not_city_covered_at_least_afternoon_(gemona_stolvizza,6).
forecasted_sky(trieste,"mostly_cloudy",winter) :- city_covered_at_least_afternoon_(trieste,1), not_city_covered_at_least_afternoon_(trieste,2).
forecasted_sky(gemona_stolvizza,"partly_cloudy",summer) :- humidity_increased_at_afternoon_neighbour(pontebba_tarvisio), not_city_covered_at_least_morning_(gemona_stolvizza,2).
forecasted_sky(pordenone,"sunny",autumn) :- wind_blowing_morning_(pordenone,"NE"), not_city_covered_at_least_morning_(pordenone,1).
forecasted_sky(pordenone,"sunny",autumn) :- not_city_covered_less_than_(pordenone,1), wind_blowing_morning_neighbour(barcis,"S").
forecasted_sky(gorizia,"mostly_cloudy",autumn) :- wind_blowing_afternoon_(gorizia,"N"), city_covered_less_than_(gorizia,6).
forecasted_rain(lignano_grado,4,spring) :- wind_blowing_afternoon_(lignano_grado,"SE"), city_covered_at_least_morning_(lignano_grado,6).
forecasted_rain(barcis,4,summer) :- humidity_decreased_at_afternoon_(barcis), not_city_covered_less_than_(barcis,5).
forecasted_sky(sappada_forni_villa,"partly_cloudy",summer) :- city_covered_at_least_afternoon_(sappada_forni_villa,3).
forecasted_rain(pordenone,2,spring) :- wind_blowing_afternoon_(pordenone,"N"), not_city_covered_at_least_afternoon_(pordenone,4).
forecasted_sky(trieste,"cloudy",spring) :- not_city_covered_less_than_(trieste,4).
forecasted_rain(gemona_stolvizza,0,winter) :- temperature_increased_at_afternoon_neighbour(sappada_forni_villa), not_city_covered_at_least_morning_(gemona_stolvizza,4).
forecasted_sky(gemona_stolvizza,"partly_cloudy",winter) :- not_city_covered_less_than_(gemona_stolvizza,2), not_city_covered_at_least_afternoon_(gemona_stolvizza,4).
forecasted_rain(gorizia,6,winter) :- city_covered_at_least_morning_(gorizia,3), wind_blowing_afternoon_(gorizia,"N").
forecasted_sky(gemona_stolvizza,"partly_cloudy",autumn) :- not_city_covered_at_least_afternoon_(gemona_stolvizza,3), temp_front_afternoon_at_300hPa(gemona_stolvizza,udine_palmanova).
forecasted_sky(pontebba_tarvisio,"mostly_clear",autumn) :- temperature_decreased_at_afternoon_(pontebba_tarvisio), humidity_decreased_at_afternoon_(pontebba_tarvisio), not_city_covered_at_least_morning_(pontebba_tarvisio,2).
forecasted_sky(gorizia,"partly_cloudy",winter) :- humidity_increased_at_afternoon_(gorizia), not_city_covered_at_least_morning_(gorizia,3).
forecasted_sky(sappada_forni_villa,"partly_cloudy",winter) :- wind_blowing_morning_neighbour(udine_palmanova,"N"), not_city_covered_at_least_afternoon_(sappada_forni_villa,4).
forecasted_sky(trieste,"cloudy",autumn) :- city_covered_at_least_afternoon_(trieste,3).
forecasted_rain(trieste,2,winter) :- city_covered_at_least_afternoon_(trieste,1), temperature_decreased_at_afternoon_(trieste), city_covered_less_than_(trieste,5).
forecasted_rain(gemona_stolvizza,0,autumn) :- not_city_covered_less_than_(gemona_stolvizza,1), city_covered_less_than_(gemona_stolvizza,4).
forecasted_sky(gemona_stolvizza,"mostly_clear",autumn) :- not_city_covered_less_than_(gemona_stolvizza,1), city_covered_less_than_(gemona_stolvizza,4).
forecasted_sky(lignano_grado,"mostly_clear",autumn) :- city_covered_less_than_(lignano_grado,2).
forecasted_sky(lignano_grado,"sunny",autumn) :- city_covered_less_than_(lignano_grado,1).
forecasted_sky(trieste,"mostly_clear",spring) :- not_city_covered_at_least_afternoon_(trieste,1), wind_blowing_morning_(trieste,"SW").
forecasted_rain(pordenone,6,autumn) :- city_covered_at_least_morning_(pordenone,3), wind_blowing_morning_(pordenone,"N").
forecasted_rain(pordenone,0,spring) :- city_covered_less_than_(pordenone,1).
forecasted_rain(lignano_grado,1,spring) :- wind_blowing_afternoon_(lignano_grado,"SE"), not_city_covered_at_least_afternoon_(lignano_grado,2).
forecasted_rain(udine_palmanova,6,spring) :- city_covered_at_least_afternoon_(udine_palmanova,6).
forecasted_rain(pordenone,2,autumn) :- not_city_covered_less_than_(pordenone,5), temperature_decreased_at_afternoon_(pordenone).
forecasted_sky(pordenone,"cloudy",winter) :- city_covered_at_least_morning_(pordenone,2), temperature_decreased_at_afternoon_(pordenone).
forecasted_rain(sappada_forni_villa,6,autumn) :- wind_blowing_afternoon_(sappada_forni_villa,"N"), not_city_covered_at_least_afternoon_(sappada_forni_villa,3).
forecasted_rain(gorizia,1,winter) :- wind_blowing_morning_(gorizia,"NE"), not_city_covered_at_least_morning_(gorizia,2).
forecasted_sky(trieste,"mostly_clear",autumn) :- not_city_covered_at_least_afternoon_(trieste,2), wind_blowing_morning_(trieste,"S").
forecasted_rain(gemona_stolvizza,1,winter) :- city_covered_at_least_morning_(gemona_stolvizza,5), wind_blowing_afternoon_neighbour(udine_palmanova,"NE").
forecasted_sky(gorizia,"mostly_clear",spring) :- city_covered_less_than_(gorizia,2), wind_blowing_afternoon_(gorizia,"S").
forecasted_sky(trieste,"sunny",autumn) :- not_city_covered_at_least_morning_(trieste,3), wind_blowing_afternoon_(trieste,"NE").
forecasted_rain(gemona_stolvizza,4,spring) :- not_city_covered_at_least_afternoon_(gemona_stolvizza,3), city_covered_at_least_morning_(gemona_stolvizza,6).
forecasted_sky(pordenone,"mostly_cloudy",winter) :- city_covered_at_least_morning_(pordenone,2), city_covered_less_than_(pordenone,5).
forecasted_rain(lignano_grado,6,spring) :- city_covered_at_least_afternoon_(lignano_grado,5).
forecasted_rain(lignano_grado,0,winter) :- temperature_increased_at_afternoon_neighbour(pordenone), not_city_covered_at_least_afternoon_(lignano_grado,4).
forecasted_sky(sappada_forni_villa,"mostly_clear",summer) :- city_covered_less_than_(sappada_forni_villa,3).
forecasted_sky(udine_palmanova,"partly_cloudy",winter) :- not_city_covered_less_than_(udine_palmanova,5), not_city_covered_at_least_morning_(udine_palmanova,3).
forecasted_sky(gemona_stolvizza,"mostly_cloudy",winter) :- city_covered_at_least_afternoon_(gemona_stolvizza,4).
forecasted_sky(pordenone,"mostly_cloudy",spring) :- temperature_increased_at_afternoon_(pordenone), not_city_covered_at_least_afternoon_(pordenone,2).
forecasted_sky(barcis,"partly_cloudy",summer) :- wind_blowing_afternoon_(barcis,"SE"), city_covered_less_than_(barcis,1).
forecasted_sky(gorizia,"mostly_clear",summer) :- not_city_covered_at_least_afternoon_(gorizia,5), temperature_increased_at_afternoon_neighbour(pontebba_tarvisio).
forecasted_rain(gorizia,0,autumn) :- not_city_covered_at_least_afternoon_(gorizia,4), wind_blowing_morning_(gorizia,"S").
forecasted_sky(pontebba_tarvisio,"partly_cloudy",autumn) :- temperature_decreased_at_afternoon_neighbour(gorizia), not_city_covered_less_than_(pontebba_tarvisio,4), not_city_covered_at_least_morning_(pontebba_tarvisio,6).
forecasted_sky(barcis,"cloudy",winter) :- city_covered_at_least_morning_(barcis,3), not_city_covered_at_least_morning_(barcis,4).
forecasted_rain(trieste,0,spring) :- humidity_increased_at_afternoon_(trieste), city_covered_less_than_(trieste,1).
forecasted_rain(lignano_grado,4,summer) :- city_covered_at_least_afternoon_(lignano_grado,2), city_covered_exactly_morning_(lignano_grado,1).
forecasted_sky(gemona_stolvizza,"cloudy",autumn) :- city_covered_at_least_morning_(gemona_stolvizza,6).
forecasted_rain(gemona_stolvizza,1,autumn) :- temperature_decreased_at_afternoon_neighbour(gorizia), city_covered_at_least_morning_(gemona_stolvizza,6).
forecasted_sky(barcis,"cloudy",autumn) :- city_covered_at_least_afternoon_(barcis,3).
forecasted_rain(gemona_stolvizza,4,autumn) :- not_city_covered_less_than_(gemona_stolvizza,6), temp_front_afternoon_at_300hPa(gemona_stolvizza,udine_palmanova).
forecasted_rain(pordenone,0,summer) :- not_city_covered_at_least_afternoon_(pordenone,1).
forecasted_rain(trieste,0,summer) :- not_city_covered_at_least_morning_(trieste,1).
forecasted_sky(gorizia,"mostly_cloudy",winter) :- city_covered_at_least_afternoon_(gorizia,1), temperature_increased_at_afternoon_(gorizia).
forecasted_sky(udine_palmanova,"cloudy",winter) :- city_covered_at_least_morning_(udine_palmanova,3).
forecasted_rain(barcis,0,autumn) :- not_city_covered_at_least_morning_(barcis,2).
forecasted_rain(gorizia,4,autumn) :- city_covered_at_least_afternoon_(gorizia,3), wind_blowing_afternoon_(gorizia,"NE").
forecasted_rain(lignano_grado,0,spring) :- city_covered_less_than_(lignano_grado,2), temp_front_afternoon_at_1000hPa(lignano_grado,pordenone).
forecasted_rain(udine_palmanova,0,autumn) :- city_covered_exactly_afternoon_(udine_palmanova,3), city_covered_at_least_morning_(udine_palmanova,5).
forecasted_sky(trieste,"sunny",autumn) :- not_city_covered_at_least_morning_(trieste,3), wind_blowing_morning_(trieste,"SW").
forecasted_rain(sappada_forni_villa,0,summer) :- city_covered_at_least_afternoon_(sappada_forni_villa,4).
forecasted_sky(sappada_forni_villa,"partly_cloudy",spring) :- city_covered_less_than_(sappada_forni_villa,2).
forecasted_sky(trieste,"mostly_clear",summer) :- city_covered_exactly_morning_(trieste,2), not_city_covered_less_than_(trieste,3).
forecasted_sky(barcis,"partly_cloudy",autumn) :- humidity_decreased_at_afternoon_(barcis), not_city_covered_less_than_(barcis,3), city_covered_less_than_(barcis,4).
forecasted_rain(udine_palmanova,4,spring) :- not_city_covered_less_than_(udine_palmanova,2), temperature_increased_at_afternoon_(udine_palmanova).
forecasted_rain(trieste,0,winter) :- not_city_covered_at_least_morning_(trieste,6), wind_blowing_afternoon_(trieste,"NE").
forecasted_rain(udine_palmanova,1,spring) :- hum_front_afternoon_at_1000hPa(udine_palmanova,gemona_stolvizza), not_city_covered_at_least_morning_(udine_palmanova,1).
forecasted_rain(barcis,1,summer) :- not_city_covered_at_least_afternoon_(barcis,3), city_covered_exactly_afternoon_(barcis,2).
forecasted_sky(gorizia,"partly_cloudy",autumn) :- wind_blowing_morning_(gorizia,"SE"), not_city_covered_at_least_morning_(gorizia,1).
forecasted_rain(udine_palmanova,4,autumn) :- hum_front_afternoon_at_500hPa(udine_palmanova,pordenone), not_city_covered_at_least_afternoon_(udine_palmanova,3).
forecasted_sky(pontebba_tarvisio,"partly_cloudy",autumn) :- city_covered_less_than_(pontebba_tarvisio,5), wind_blowing_afternoon_(pontebba_tarvisio,"SE").
forecasted_sky(lignano_grado,"partly_cloudy",autumn) :- city_covered_at_least_morning_(lignano_grado,5), wind_blowing_afternoon_neighbour(trieste,"NE").
forecasted_rain(gorizia,4,summer) :- city_covered_at_least_morning_(gorizia,2), wind_blowing_afternoon_(gorizia,"E").
forecasted_sky(pordenone,"cloudy",spring) :- city_covered_at_least_morning_(pordenone,4).
forecasted_rain(pontebba_tarvisio,6,autumn) :- wind_blowing_afternoon_(pontebba_tarvisio,"N"), city_covered_at_least_morning_(pontebba_tarvisio,4).
forecasted_sky(gemona_stolvizza,"partly_cloudy",spring) :- city_covered_less_than_(gemona_stolvizza,3).
forecasted_sky(trieste,"mostly_clear",autumn) :- wind_blowing_afternoon_(trieste,"E"), city_covered_less_than_(trieste,2).
forecasted_sky(gorizia,"partly_cloudy",summer) :- wind_blowing_morning_(gorizia,"E"), city_covered_less_than_(gorizia,4).
forecasted_sky(trieste,"cloudy",winter) :- not_city_covered_less_than_(trieste,6), city_covered_at_least_morning_(trieste,3).
forecasted_rain(barcis,0,summer) :- not_city_covered_at_least_afternoon_(barcis,6), not_city_covered_at_least_morning_(barcis,2).
forecasted_sky(sappada_forni_villa,"mostly_cloudy",spring) :- city_covered_at_least_afternoon_(sappada_forni_villa,3).
forecasted_rain(pontebba_tarvisio,0,summer) :- not_city_covered_at_least_morning_(pontebba_tarvisio,3).
forecasted_sky(gorizia,"partly_cloudy",autumn) :- not_city_covered_less_than_(gorizia,1), temperature_decreased_at_afternoon_(gorizia), wind_blowing_morning_(gorizia,"NE").
forecasted_rain(sappada_forni_villa,4,autumn) :- wind_blowing_morning_neighbour(udine_palmanova,"NE"), city_covered_at_least_afternoon_(sappada_forni_villa,4).
forecasted_rain(gorizia,2,winter) :- not_city_covered_less_than_(gorizia,6), humidity_increased_at_afternoon_neighbour(pontebba_tarvisio).
forecasted_rain(pordenone,4,spring) :- city_covered_at_least_afternoon_(pordenone,6).
forecasted_sky(barcis,"cloudy",spring) :- not_city_covered_less_than_(barcis,1).
forecasted_sky(udine_palmanova,"mostly_clear",autumn) :- not_city_covered_at_least_morning_(udine_palmanova,3), hum_front_afternoon_at_1000hPa(udine_palmanova,gemona_stolvizza).
forecasted_rain(gorizia,0,autumn) :- wind_blowing_morning_(gorizia,"E"), not_city_covered_at_least_afternoon_(gorizia,2).
forecasted_rain(trieste,1,autumn) :- city_covered_at_least_morning_(trieste,3), wind_blowing_morning_(trieste,"E"), not_city_covered_at_least_afternoon_(trieste,4).
forecasted_sky(barcis,"partly_cloudy",autumn) :- wind_blowing_afternoon_(barcis,"E"), not_city_covered_at_least_afternoon_(barcis,1), temp_front_morning_at_1000hPa(barcis,pordenone).
forecasted_sky(trieste,"sunny",summer) :- city_covered_less_than_(trieste,1).
forecasted_rain(barcis,2,summer) :- city_covered_at_least_morning_(barcis,2), temperature_decreased_at_afternoon_(barcis).
forecasted_rain(pontebba_tarvisio,1,spring) :- city_covered_at_least_afternoon_(pontebba_tarvisio,1), wind_blowing_morning_neighbour(gorizia,"E").
forecasted_rain(pontebba_tarvisio,0,spring) :- city_covered_less_than_(pontebba_tarvisio,6), wind_blowing_morning_(pontebba_tarvisio,"NE").
forecasted_sky(sappada_forni_villa,"mostly_clear",winter) :- not_city_covered_at_least_morning_(sappada_forni_villa,1), temp_front_afternoon_at_300hPa(sappada_forni_villa,udine_palmanova).
forecasted_rain(udine_palmanova,6,autumn) :- city_covered_at_least_morning_(udine_palmanova,2), city_covered_less_than_(udine_palmanova,3).
forecasted_rain(sappada_forni_villa,1,winter) :- city_covered_less_than_(sappada_forni_villa,3), wind_blowing_morning_(sappada_forni_villa,"E").
forecasted_rain(pontebba_tarvisio,2,autumn) :- city_covered_at_least_morning_(pontebba_tarvisio,4), temp_front_afternoon_at_1000hPa(pontebba_tarvisio,sappada_forni_villa).
forecasted_rain(gemona_stolvizza,2,winter) :- city_covered_at_least_morning_(gemona_stolvizza,4), wind_blowing_morning_(gemona_stolvizza,"E").
forecasted_rain(pordenone,1,autumn) :- city_covered_at_least_morning_(pordenone,3), temp_front_afternoon_at_1000hPa(pordenone,sappada_forni_villa).
forecasted_sky(pordenone,"partly_cloudy",summer) :- city_covered_at_least_morning_(pordenone,1), city_covered_at_least_afternoon_(pordenone,1).
forecasted_sky(pordenone,"partly_cloudy",winter) :- not_city_covered_at_least_afternoon_(pordenone,2), hum_front_afternoon_at_1000hPa(pordenone,barcis).
forecasted_rain(udine_palmanova,0,spring) :- temperature_decreased_at_afternoon_(udine_palmanova), not_city_covered_at_least_morning_(udine_palmanova,4).
forecasted_rain(pordenone,0,spring) :- not_city_covered_at_least_morning_(pordenone,5), city_covered_exactly_morning_(pordenone,2).
forecasted_rain(pontebba_tarvisio,2,winter) :- wind_blowing_morning_(pontebba_tarvisio,"N"), not_city_covered_at_least_morning_(pontebba_tarvisio,3).
forecasted_rain(lignano_grado,0,winter) :- city_covered_less_than_(lignano_grado,4).
forecasted_sky(lignano_grado,"mostly_cloudy",autumn) :- not_city_covered_less_than_(lignano_grado,5), temperature_increased_at_afternoon_(lignano_grado).
forecasted_sky(trieste,"mostly_cloudy",winter) :- city_covered_at_least_morning_(trieste,1), wind_blowing_afternoon_(trieste,"E").
forecasted_rain(lignano_grado,0,autumn) :- not_city_covered_at_least_morning_(lignano_grado,5), wind_blowing_morning_(lignano_grado,"S").
forecasted_rain(udine_palmanova,2,autumn) :- humidity_increased_at_afternoon_(udine_palmanova), not_city_covered_less_than_(udine_palmanova,4), city_covered_less_than_(udine_palmanova,5).
forecasted_sky(gorizia,"partly_cloudy",autumn) :- wind_blowing_afternoon_(gorizia,"E"), city_covered_less_than_(gorizia,4).
forecasted_sky(barcis,"partly_cloudy",winter) :- temperature_increased_at_afternoon_neighbour(sappada_forni_villa), not_city_covered_at_least_afternoon_(barcis,1).
forecasted_sky(gorizia,"partly_cloudy",winter) :- not_city_covered_less_than_(gorizia,1), city_covered_less_than_(gorizia,3).
forecasted_rain(gemona_stolvizza,0,summer) :- city_covered_at_least_afternoon_(gemona_stolvizza,2), humidity_decreased_at_afternoon_(gemona_stolvizza).
forecasted_rain(gorizia,0,summer) :- not_city_covered_at_least_morning_(gorizia,2).
forecasted_rain(lignano_grado,2,winter) :- city_covered_at_least_afternoon_(lignano_grado,1), temperature_decreased_at_afternoon_(lignano_grado), not_city_covered_at_least_morning_(lignano_grado,4).
forecasted_sky(pordenone,"mostly_clear",winter) :- not_city_covered_at_least_morning_(pordenone,2), wind_blowing_morning_(pordenone,"S").
forecasted_sky(gorizia,"partly_cloudy",spring) :- city_covered_less_than_(gorizia,2), temp_front_afternoon_at_700hPa(gorizia,pontebba_tarvisio).
forecasted_sky(sappada_forni_villa,"mostly_cloudy",autumn) :- temperature_decreased_at_afternoon_(sappada_forni_villa), not_city_covered_less_than_(sappada_forni_villa,4).
forecasted_sky(gemona_stolvizza,"cloudy",winter) :- city_covered_at_least_afternoon_(gemona_stolvizza,4).
forecasted_rain(gorizia,1,autumn) :- not_city_covered_less_than_(gorizia,6), temperature_decreased_at_afternoon_(gorizia), humidity_increased_at_afternoon_(gorizia).
forecasted_rain(pontebba_tarvisio,0,winter) :- city_covered_at_least_morning_(pontebba_tarvisio,3), not_city_covered_at_least_morning_(pontebba_tarvisio,5).
forecasted_sky(lignano_grado,"mostly_clear",summer) :- not_city_covered_at_least_afternoon_(lignano_grado,1).
forecasted_rain(udine_palmanova,2,spring) :- wind_blowing_afternoon_neighbour(pordenone,"N"), not_city_covered_at_least_morning_(udine_palmanova,6).
forecasted_sky(sappada_forni_villa,"cloudy",spring) :- not_city_covered_less_than_(sappada_forni_villa,6).
forecasted_rain(pontebba_tarvisio,0,autumn) :- city_covered_less_than_(pontebba_tarvisio,1).
forecasted_sky(trieste,"partly_cloudy",winter) :- humidity_increased_at_afternoon_(trieste), not_city_covered_at_least_morning_(trieste,3).
forecasted_sky(trieste,"mostly_cloudy",spring) :- city_covered_at_least_afternoon_(trieste,4).
forecasted_sky(trieste,"mostly_clear",autumn) :- wind_blowing_afternoon_(trieste,"NE"), city_covered_less_than_(trieste,2).
forecasted_rain(gorizia,0,winter) :- city_covered_at_least_morning_(gorizia,4), hum_front_morning_at_500hPa(gorizia,gemona_stolvizza).
forecasted_rain(udine_palmanova,4,winter) :- wind_blowing_afternoon_(udine_palmanova,"N"), city_covered_at_least_afternoon_(udine_palmanova,4).
forecasted_rain(gorizia,1,winter) :- city_covered_at_least_morning_(gorizia,4), wind_blowing_morning_(gorizia,"NE").
forecasted_rain(trieste,4,summer) :- city_covered_at_least_morning_(trieste,4).
forecasted_sky(lignano_grado,"mostly_clear",autumn) :- city_covered_at_least_morning_(lignano_grado,4), wind_blowing_afternoon_(lignano_grado,"S").
forecasted_rain(barcis,0,spring) :- not_city_covered_at_least_afternoon_(barcis,4), wind_blowing_morning_(barcis,"NE").
forecasted_sky(udine_palmanova,"mostly_clear",spring) :- not_city_covered_at_least_morning_(udine_palmanova,1), wind_blowing_afternoon_(udine_palmanova,"S").
forecasted_rain(gorizia,4,winter) :- city_covered_at_least_morning_(gorizia,5), temp_front_afternoon_at_300hPa(gorizia,pontebba_tarvisio).
forecasted_sky(pordenone,"partly_cloudy",autumn) :- temperature_decreased_at_afternoon_(pordenone), not_city_covered_less_than_(pordenone,6).
forecasted_sky(pontebba_tarvisio,"cloudy",winter) :- not_city_covered_less_than_(pontebba_tarvisio,3), not_city_covered_at_least_morning_(pontebba_tarvisio,3).
forecasted_sky(udine_palmanova,"sunny",autumn) :- not_city_covered_at_least_morning_(udine_palmanova,1).
forecasted_rain(lignano_grado,2,spring) :- not_city_covered_less_than_(lignano_grado,2), not_city_covered_at_least_morning_(lignano_grado,6).
forecasted_rain(pontebba_tarvisio,6,spring) :- temp_front_afternoon_at_500hPa(pontebba_tarvisio,sappada_forni_villa), city_covered_at_least_afternoon_(pontebba_tarvisio,6).
forecasted_sky(sappada_forni_villa,"cloudy",winter) :- not_city_covered_less_than_(sappada_forni_villa,6).
forecasted_sky(udine_palmanova,"mostly_clear",summer) :- not_city_covered_at_least_afternoon_(udine_palmanova,1), hum_front_morning_at_500hPa(udine_palmanova,sappada_forni_villa).
forecasted_sky(udine_palmanova,"mostly_clear",summer) :- city_covered_at_least_afternoon_(udine_palmanova,3).
forecasted_sky(trieste,"partly_cloudy",autumn) :- wind_blowing_afternoon_(trieste,"NE"), city_covered_at_least_morning_(trieste,6).
forecasted_sky(udine_palmanova,"sunny",summer) :- humidity_decreased_at_afternoon_neighbour(sappada_forni_villa), city_covered_less_than_(udine_palmanova,4).
forecasted_sky(sappada_forni_villa,"partly_cloudy",autumn) :- not_city_covered_at_least_morning_(sappada_forni_villa,2), wind_blowing_morning_(sappada_forni_villa,"S").
forecasted_sky(udine_palmanova,"partly_cloudy",autumn) :- temperature_decreased_at_afternoon_(udine_palmanova), city_covered_at_least_morning_(udine_palmanova,3), not_city_covered_at_least_afternoon_(udine_palmanova,3).
forecasted_sky(gorizia,"cloudy",winter) :- city_covered_at_least_morning_(gorizia,3).
forecasted_rain(udine_palmanova,0,spring) :- not_city_covered_at_least_morning_(udine_palmanova,3), hum_front_afternoon_at_500hPa(udine_palmanova,pordenone).
forecasted_sky(lignano_grado,"sunny",summer) :- city_covered_less_than_(lignano_grado,3).
forecasted_rain(udine_palmanova,0,winter) :- not_city_covered_at_least_morning_(udine_palmanova,2).
forecasted_rain(gemona_stolvizza,0,spring) :- city_covered_less_than_(gemona_stolvizza,5).
forecasted_rain(barcis,2,autumn) :- not_city_covered_less_than_(barcis,6), wind_blowing_morning_(barcis,"E").
forecasted_sky(gemona_stolvizza,"mostly_cloudy",spring) :- city_covered_at_least_afternoon_(gemona_stolvizza,1).
forecasted_rain(sappada_forni_villa,1,autumn) :- city_covered_less_than_(sappada_forni_villa,6), city_covered_at_least_morning_(sappada_forni_villa,3).
forecasted_sky(sappada_forni_villa,"mostly_clear",autumn) :- not_city_covered_at_least_afternoon_(sappada_forni_villa,2), wind_blowing_afternoon_(sappada_forni_villa,"NE").
forecasted_rain(pontebba_tarvisio,0,winter) :- city_covered_less_than_(pontebba_tarvisio,3).
forecasted_sky(barcis,"cloudy",winter) :- city_covered_at_least_morning_(barcis,5).
forecasted_rain(sappada_forni_villa,4,spring) :- not_city_covered_at_least_afternoon_(sappada_forni_villa,5), city_covered_at_least_morning_(sappada_forni_villa,3).
forecasted_sky(trieste,"cloudy",winter) :- temperature_decreased_at_afternoon_(trieste), city_covered_less_than_(trieste,4).
forecasted_sky(gorizia,"sunny",autumn) :- wind_blowing_afternoon_(gorizia,"NE"), city_covered_less_than_(gorizia,2).
forecasted_rain(sappada_forni_villa,6,winter) :- not_city_covered_less_than_(sappada_forni_villa,5), temp_front_afternoon_at_700hPa(sappada_forni_villa,udine_palmanova).
forecasted_rain(lignano_grado,4,winter) :- city_covered_at_least_afternoon_(lignano_grado,4), temp_front_morning_at_1000hPa(lignano_grado,trieste).
forecasted_rain(gorizia,0,autumn) :- wind_blowing_afternoon_(gorizia,"NE"), not_city_covered_at_least_morning_(gorizia,2).
forecasted_sky(sappada_forni_villa,"mostly_clear",autumn) :- hum_front_afternoon_at_1000hPa(sappada_forni_villa,pordenone), not_city_covered_less_than_(sappada_forni_villa,5).
forecasted_rain(pordenone,0,summer) :- not_city_covered_at_least_morning_(pordenone,1).
forecasted_rain(gorizia,0,autumn) :- not_city_covered_less_than_(gorizia,2), temp_front_morning_at_1000hPa(gorizia,pontebba_tarvisio).
forecasted_rain(trieste,0,autumn) :- humidity_increased_at_afternoon_(trieste), city_covered_less_than_(trieste,2).
forecasted_sky(barcis,"mostly_cloudy",winter) :- not_city_covered_at_least_morning_(barcis,1), wind_blowing_morning_(barcis,"NE").
forecasted_rain(pordenone,4,winter) :- city_covered_at_least_morning_(pordenone,5).
forecasted_rain(gorizia,1,spring) :- not_city_covered_less_than_(gorizia,1), not_city_covered_at_least_morning_(gorizia,2).
forecasted_sky(barcis,"mostly_clear",autumn) :- not_city_covered_at_least_afternoon_(barcis,3), temp_front_morning_at_500hPa(barcis,sappada_forni_villa).
forecasted_sky(gemona_stolvizza,"mostly_cloudy",autumn) :- city_covered_at_least_morning_(gemona_stolvizza,3), city_covered_less_than_(gemona_stolvizza,6).
forecasted_rain(trieste,4,autumn) :- wind_blowing_afternoon_(trieste,"N"), not_city_covered_at_least_afternoon_(trieste,3).
forecasted_sky(lignano_grado,"partly_cloudy",winter) :- not_city_covered_at_least_morning_(lignano_grado,1).
forecasted_rain(barcis,1,autumn) :- not_city_covered_at_least_afternoon_(barcis,4), wind_blowing_afternoon_(barcis,"NE"), city_covered_exactly_morning_(barcis,2).
forecasted_rain(trieste,0,autumn) :- not_city_covered_less_than_(trieste,3), temperature_increased_at_afternoon_(trieste), not_city_covered_at_least_afternoon_(trieste,3).
forecasted_rain(pontebba_tarvisio,2,spring) :- wind_blowing_afternoon_(pontebba_tarvisio,"N"), not_city_covered_at_least_afternoon_(pontebba_tarvisio,6).
forecasted_rain(sappada_forni_villa,2,summer) :- not_city_covered_at_least_afternoon_(sappada_forni_villa,4), not_city_covered_less_than_(sappada_forni_villa,3).
forecasted_sky(barcis,"mostly_cloudy",summer) :- humidity_decreased_at_afternoon_(barcis), not_city_covered_less_than_(barcis,5).
forecasted_rain(trieste,0,spring) :- city_covered_exactly_afternoon_(trieste,1), not_city_covered_at_least_morning_(trieste,1).
forecasted_rain(barcis,4,winter) :- city_covered_at_least_afternoon_(barcis,3), city_covered_exactly_morning_(barcis,5).
forecasted_sky(pontebba_tarvisio,"mostly_clear",autumn) :- wind_blowing_morning_neighbour(gorizia,"E"), not_city_covered_at_least_afternoon_(pontebba_tarvisio,1).
forecasted_sky(sappada_forni_villa,"partly_cloudy",autumn) :- not_city_covered_at_least_morning_(sappada_forni_villa,2), hum_front_morning_at_1000hPa(sappada_forni_villa,pordenone).
forecasted_rain(barcis,0,autumn) :- not_city_covered_at_least_morning_(barcis,4), wind_blowing_morning_(barcis,"S").
forecasted_sky(barcis,"mostly_clear",autumn) :- city_covered_less_than_(barcis,6), wind_blowing_morning_(barcis,"S").
forecasted_rain(trieste,4,winter) :- city_covered_at_least_morning_(trieste,3), hum_front_afternoon_at_500hPa(trieste,lignano_grado).
forecasted_sky(sappada_forni_villa,"cloudy",autumn) :- city_covered_at_least_morning_(sappada_forni_villa,1), wind_blowing_morning_(sappada_forni_villa,"E").
forecasted_sky(lignano_grado,"mostly_cloudy",spring) :- not_city_covered_less_than_(lignano_grado,1).
forecasted_sky(gorizia,"mostly_clear",autumn) :- city_covered_at_least_morning_(gorizia,1), city_covered_at_least_afternoon_(gorizia,1), city_covered_less_than_(gorizia,5).
forecasted_sky(trieste,"mostly_clear",spring) :- not_city_covered_at_least_afternoon_(trieste,1), hum_front_morning_at_1000hPa(trieste,lignano_grado).
forecasted_rain(gemona_stolvizza,6,winter) :- wind_blowing_afternoon_neighbour(sappada_forni_villa,"N"), city_covered_at_least_morning_(gemona_stolvizza,4).
forecasted_sky(pordenone,"mostly_clear",winter) :- not_city_covered_at_least_morning_(pordenone,4), wind_blowing_afternoon_(pordenone,"NE").
forecasted_sky(gorizia,"mostly_cloudy",spring) :- not_city_covered_less_than_(gorizia,1).
forecasted_rain(gemona_stolvizza,6,winter) :- wind_blowing_morning_(gemona_stolvizza,"N"), city_covered_at_least_morning_(gemona_stolvizza,5).
forecasted_rain(gorizia,2,spring) :- not_city_covered_less_than_(gorizia,4), not_city_covered_at_least_morning_(gorizia,5).
forecasted_rain(lignano_grado,0,summer) :- city_covered_less_than_(lignano_grado,3).
forecasted_sky(gemona_stolvizza,"mostly_clear",summer) :- city_covered_at_least_morning_(gemona_stolvizza,2), city_covered_less_than_(gemona_stolvizza,4).
forecasted_rain(sappada_forni_villa,0,winter) :- not_city_covered_at_least_afternoon_(sappada_forni_villa,2), wind_blowing_afternoon_(sappada_forni_villa,"SE").
forecasted_rain(barcis,4,spring) :- city_covered_at_least_afternoon_(barcis,4), temperature_increased_at_afternoon_(barcis), wind_blowing_morning_(barcis,"N").
forecasted_rain(gemona_stolvizza,1,summer) :- city_covered_at_least_afternoon_(gemona_stolvizza,4), not_city_covered_at_least_morning_(gemona_stolvizza,3).
forecasted_rain(gemona_stolvizza,1,spring) :- not_city_covered_at_least_morning_(gemona_stolvizza,3), city_covered_exactly_afternoon_(gemona_stolvizza,5).
forecasted_rain(pordenone,4,summer) :- city_covered_at_least_morning_(pordenone,1), city_covered_at_least_afternoon_(pordenone,1).
forecasted_rain(barcis,1,spring) :- humidity_decreased_at_afternoon_(barcis), not_city_covered_at_least_afternoon_(barcis,4).
forecasted_sky(gorizia,"mostly_cloudy",winter) :- wind_blowing_afternoon_(gorizia,"E"), city_covered_less_than_(gorizia,1).
forecasted_rain(gemona_stolvizza,1,autumn) :- humidity_decreased_at_afternoon_(gemona_stolvizza), city_covered_less_than_(gemona_stolvizza,1).
forecasted_sky(lignano_grado,"mostly_clear",spring) :- humidity_increased_at_afternoon_neighbour(trieste), not_city_covered_at_least_morning_(lignano_grado,2).
forecasted_sky(barcis,"partly_cloudy",spring) :- not_city_covered_at_least_afternoon_(barcis,2).
forecasted_rain(pontebba_tarvisio,2,summer) :- not_city_covered_less_than_(pontebba_tarvisio,3), humidity_increased_at_afternoon_(pontebba_tarvisio).
forecasted_rain(udine_palmanova,1,autumn) :- city_covered_at_least_morning_(udine_palmanova,4), humidity_decreased_at_afternoon_(udine_palmanova).
forecasted_rain(udine_palmanova,6,winter) :- not_city_covered_at_least_morning_(udine_palmanova,5), hum_front_morning_at_500hPa(udine_palmanova,pordenone).
forecasted_rain(trieste,6,autumn) :- wind_blowing_morning_(trieste,"NE"), city_covered_at_least_morning_(trieste,6).
forecasted_rain(pordenone,2,autumn) :- temp_front_afternoon_at_700hPa(pordenone,udine_palmanova), city_covered_at_least_morning_(pordenone,6).
forecasted_sky(udine_palmanova,"mostly_cloudy",spring) :- not_city_covered_at_least_morning_(udine_palmanova,1), hum_front_morning_at_500hPa(udine_palmanova,sappada_forni_villa).
forecasted_rain(pordenone,0,winter) :- not_city_covered_at_least_morning_(pordenone,2).
forecasted_rain(barcis,1,winter) :- city_covered_at_least_morning_(barcis,3), not_city_covered_at_least_morning_(barcis,4).
forecasted_sky(barcis,"sunny",summer) :- wind_blowing_morning_(barcis,"E"), not_city_covered_at_least_afternoon_(barcis,1).
forecasted_sky(gemona_stolvizza,"cloudy",autumn) :- not_city_covered_less_than_(gemona_stolvizza,4), wind_blowing_morning_(gemona_stolvizza,"N").
forecasted_rain(lignano_grado,0,spring) :- temperature_decreased_at_afternoon_neighbour(udine_palmanova), city_covered_less_than_(lignano_grado,2).
forecasted_rain(lignano_grado,6,autumn) :- humidity_increased_at_afternoon_neighbour(trieste), not_city_covered_less_than_(lignano_grado,5).
forecasted_rain(pordenone,0,autumn) :- humidity_decreased_at_afternoon_(pordenone), not_city_covered_at_least_afternoon_(pordenone,6), wind_blowing_morning_(pordenone,"E").
forecasted_sky(sappada_forni_villa,"partly_cloudy",summer) :- not_city_covered_less_than_(sappada_forni_villa,6).
forecasted_sky(gorizia,"cloudy",autumn) :- city_covered_at_least_morning_(gorizia,6).
forecasted_rain(gorizia,0,winter) :- city_covered_less_than_(gorizia,5), temperature_increased_at_afternoon_neighbour(pontebba_tarvisio).
forecasted_rain(pontebba_tarvisio,0,spring) :- not_city_covered_at_least_morning_(pontebba_tarvisio,1).
forecasted_rain(gorizia,6,autumn) :- humidity_increased_at_afternoon_neighbour(udine_palmanova), city_covered_at_least_morning_(gorizia,5).
forecasted_sky(pontebba_tarvisio,"partly_cloudy",winter) :- not_city_covered_less_than_(pontebba_tarvisio,1), not_city_covered_at_least_afternoon_(pontebba_tarvisio,1).
forecasted_rain(sappada_forni_villa,0,autumn) :- city_covered_less_than_(sappada_forni_villa,2).
forecasted_rain(pordenone,0,winter) :- temperature_increased_at_afternoon_(pordenone), not_city_covered_at_least_morning_(pordenone,5).
forecasted_sky(udine_palmanova,"mostly_clear",autumn) :- not_city_covered_at_least_morning_(udine_palmanova,1).
forecasted_rain(sappada_forni_villa,2,autumn) :- not_city_covered_at_least_afternoon_(sappada_forni_villa,4), wind_blowing_afternoon_neighbour(gemona_stolvizza,"NE"), city_covered_at_least_afternoon_(sappada_forni_villa,2).
forecasted_sky(pordenone,"mostly_clear",autumn) :- wind_blowing_morning_(pordenone,"NE"), not_city_covered_at_least_afternoon_(pordenone,1).
forecasted_sky(udine_palmanova,"partly_cloudy",autumn) :- city_covered_at_least_morning_(udine_palmanova,1), city_covered_less_than_(udine_palmanova,2).
forecasted_sky(pordenone,"sunny",summer) :- city_covered_less_than_(pordenone,2).
forecasted_rain(sappada_forni_villa,0,winter) :- humidity_decreased_at_afternoon_(sappada_forni_villa), not_city_covered_at_least_morning_(sappada_forni_villa,3).
forecasted_rain(pontebba_tarvisio,0,autumn) :- city_covered_at_least_afternoon_(pontebba_tarvisio,3), not_city_covered_at_least_afternoon_(pontebba_tarvisio,5).
forecasted_rain(gemona_stolvizza,0,summer) :- not_city_covered_at_least_afternoon_(gemona_stolvizza,1).
forecasted_rain(sappada_forni_villa,0,autumn) :- wind_blowing_afternoon_neighbour(udine_palmanova,"E"), city_covered_at_least_afternoon_(sappada_forni_villa,4).
forecasted_sky(lignano_grado,"cloudy",winter) :- city_covered_at_least_afternoon_(lignano_grado,3).
forecasted_rain(udine_palmanova,0,winter) :- city_covered_at_least_morning_(udine_palmanova,2), wind_blowing_afternoon_(udine_palmanova,"NE").
forecasted_sky(udine_palmanova,"cloudy",spring) :- not_city_covered_less_than_(udine_palmanova,2).
forecasted_sky(pontebba_tarvisio,"partly_cloudy",autumn) :- not_city_covered_at_least_morning_(pontebba_tarvisio,1), hum_front_morning_at_700hPa(pontebba_tarvisio,gorizia).
forecasted_rain(udine_palmanova,2,winter) :- temperature_decreased_at_afternoon_(udine_palmanova), city_covered_at_least_morning_(udine_palmanova,4).
forecasted_rain(pordenone,0,autumn) :- temperature_increased_at_afternoon_neighbour(barcis), not_city_covered_at_least_morning_(pordenone,3).
forecasted_sky(udine_palmanova,"partly_cloudy",summer) :- city_covered_at_least_morning_(udine_palmanova,3).
forecasted_sky(sappada_forni_villa,"cloudy",autumn) :- city_covered_at_least_morning_(sappada_forni_villa,6).
forecasted_sky(barcis,"partly_cloudy",winter) :- city_covered_at_least_afternoon_(barcis,2), wind_blowing_morning_(barcis,"NW").
forecasted_sky(gorizia,"mostly_cloudy",autumn) :- humidity_decreased_at_afternoon_(gorizia), city_covered_at_least_afternoon_(gorizia,5).
forecasted_rain(pontebba_tarvisio,6,winter) :- wind_blowing_afternoon_(pontebba_tarvisio,"N"), not_city_covered_at_least_morning_(pontebba_tarvisio,2).
forecasted_sky(lignano_grado,"partly_cloudy",spring) :- wind_blowing_afternoon_(lignano_grado,"SE"), not_city_covered_at_least_afternoon_(lignano_grado,2).
forecasted_sky(sappada_forni_villa,"partly_cloudy",winter) :- temp_front_afternoon_at_1000hPa(sappada_forni_villa,udine_palmanova), city_covered_less_than_(sappada_forni_villa,1).
forecasted_rain(pontebba_tarvisio,1,autumn) :- city_covered_at_least_afternoon_(pontebba_tarvisio,1), not_city_covered_at_least_morning_(pontebba_tarvisio,4), wind_blowing_afternoon_(pontebba_tarvisio,"NE").
forecasted_sky(pordenone,"partly_cloudy",spring) :- not_city_covered_less_than_(pordenone,2), not_city_covered_at_least_morning_(pordenone,3).
forecasted_rain(pontebba_tarvisio,0,summer) :- city_covered_at_least_morning_(pontebba_tarvisio,6).
forecasted_rain(barcis,0,autumn) :- city_covered_less_than_(barcis,5), wind_blowing_morning_(barcis,"SE").
forecasted_sky(pontebba_tarvisio,"mostly_clear",summer) :- city_covered_at_least_morning_(pontebba_tarvisio,6).
forecasted_sky(pordenone,"mostly_clear",autumn) :- humidity_decreased_at_afternoon_(pordenone), city_covered_less_than_(pordenone,2).
forecasted_sky(pontebba_tarvisio,"mostly_cloudy",autumn) :- city_covered_at_least_morning_(pontebba_tarvisio,1), city_covered_less_than_(pontebba_tarvisio,4).
forecasted_rain(udine_palmanova,0,summer) :- city_covered_less_than_(udine_palmanova,5).
forecasted_sky(barcis,"mostly_clear",winter) :- city_covered_at_least_afternoon_(barcis,2), wind_blowing_morning_(barcis,"S").
forecasted_rain(pordenone,1,winter) :- not_city_covered_less_than_(pordenone,5), wind_blowing_morning_(pordenone,"E").
forecasted_sky(gorizia,"mostly_clear",winter) :- wind_blowing_afternoon_(gorizia,"NE"), not_city_covered_at_least_morning_(gorizia,2).
forecasted_rain(sappada_forni_villa,0,summer) :- city_covered_less_than_(sappada_forni_villa,3).
forecasted_sky(trieste,"partly_cloudy",spring) :- humidity_decreased_at_afternoon_(trieste), city_covered_less_than_(trieste,3).
forecasted_sky(gorizia,"mostly_clear",autumn) :- not_city_covered_at_least_morning_(gorizia,1), wind_blowing_morning_(gorizia,"S").
forecasted_rain(pontebba_tarvisio,4,autumn) :- wind_blowing_afternoon_(pontebba_tarvisio,"N"), not_city_covered_at_least_afternoon_(pontebba_tarvisio,2).
forecasted_sky(lignano_grado,"mostly_clear",winter) :- not_city_covered_at_least_morning_(lignano_grado,2), city_covered_exactly_morning_(lignano_grado,1).
forecasted_rain(pontebba_tarvisio,0,autumn) :- not_city_covered_at_least_morning_(pontebba_tarvisio,4), wind_blowing_afternoon_(pontebba_tarvisio,"E").
forecasted_sky(sappada_forni_villa,"cloudy",winter) :- not_city_covered_less_than_(sappada_forni_villa,1), temperature_decreased_at_afternoon_(sappada_forni_villa).
forecasted_sky(udine_palmanova,"partly_cloudy",winter) :- city_covered_at_least_morning_(udine_palmanova,1), city_covered_less_than_(udine_palmanova,3).
forecasted_rain(trieste,0,autumn) :- wind_blowing_afternoon_(trieste,"E"), city_covered_less_than_(trieste,2).
forecasted_sky(trieste,"mostly_cloudy",autumn) :- not_city_covered_less_than_(trieste,6), wind_blowing_afternoon_neighbour(lignano_grado,"NE").
forecasted_rain(trieste,6,autumn) :- city_covered_at_least_afternoon_(trieste,5).
forecasted_sky(pordenone,"mostly_clear",summer) :- not_city_covered_at_least_morning_(pordenone,1).
forecasted_sky(trieste,"partly_cloudy",summer) :- not_city_covered_less_than_(trieste,1), city_covered_exactly_morning_(trieste,1).
forecasted_rain(gorizia,4,spring) :- not_city_covered_less_than_(gorizia,5), temp_front_morning_at_1000hPa(gorizia,pontebba_tarvisio).
forecasted_sky(barcis,"cloudy",autumn) :- city_covered_at_least_morning_(barcis,6).
forecasted_sky(pontebba_tarvisio,"sunny",summer) :- not_city_covered_at_least_morning_(pontebba_tarvisio,2).
forecasted_sky(lignano_grado,"mostly_cloudy",winter) :- city_covered_at_least_morning_(lignano_grado,5).
forecasted_rain(pontebba_tarvisio,4,spring) :- city_covered_at_least_morning_(pontebba_tarvisio,5), temp_front_morning_at_1000hPa(pontebba_tarvisio,sappada_forni_villa).
forecasted_sky(gorizia,"cloudy",autumn) :- city_covered_at_least_afternoon_(gorizia,4), temp_front_afternoon_at_500hPa(gorizia,pontebba_tarvisio).
forecasted_sky(trieste,"partly_cloudy",summer) :- city_covered_at_least_morning_(trieste,4).
forecasted_sky(trieste,"mostly_cloudy",spring) :- city_covered_at_least_morning_(trieste,1).
forecasted_rain(lignano_grado,6,winter) :- wind_blowing_morning_(lignano_grado,"NE"), city_covered_at_least_afternoon_(lignano_grado,4).
forecasted_rain(lignano_grado,2,summer) :- not_city_covered_less_than_(lignano_grado,4).
forecasted_sky(pordenone,"cloudy",autumn) :- city_covered_at_least_morning_(pordenone,5).
forecasted_sky(trieste,"partly_cloudy",spring) :- wind_blowing_morning_(trieste,"E"), city_covered_less_than_(trieste,1).
forecasted_rain(gemona_stolvizza,4,autumn) :- wind_blowing_afternoon_(gemona_stolvizza,"NE"), city_covered_at_least_afternoon_(gemona_stolvizza,5).
forecasted_sky(gemona_stolvizza,"mostly_cloudy",winter) :- city_covered_less_than_(gemona_stolvizza,2).
forecasted_sky(sappada_forni_villa,"mostly_clear",autumn) :- wind_blowing_morning_(sappada_forni_villa,"E"), city_covered_less_than_(sappada_forni_villa,1).
forecasted_rain(sappada_forni_villa,0,spring) :- city_covered_at_least_afternoon_(sappada_forni_villa,3), wind_blowing_afternoon_(sappada_forni_villa,"NE").
forecasted_sky(barcis,"mostly_cloudy",winter) :- city_covered_at_least_morning_(barcis,4), humidity_increased_at_afternoon_(barcis).
forecasted_sky(pontebba_tarvisio,"mostly_cloudy",spring) :- not_city_covered_less_than_(pontebba_tarvisio,1), hum_front_afternoon_at_500hPa(pontebba_tarvisio,gorizia).
forecasted_sky(udine_palmanova,"mostly_cloudy",spring) :- city_covered_at_least_morning_(udine_palmanova,3).
forecasted_sky(sappada_forni_villa,"sunny",summer) :- not_city_covered_at_least_afternoon_(sappada_forni_villa,1).
forecasted_sky(pontebba_tarvisio,"mostly_clear",summer) :- not_city_covered_at_least_morning_(pontebba_tarvisio,3).
forecasted_rain(sappada_forni_villa,2,spring) :- hum_front_afternoon_at_1000hPa(sappada_forni_villa,pordenone), city_covered_at_least_afternoon_(sappada_forni_villa,5).
forecasted_rain(sappada_forni_villa,4,winter) :- city_covered_at_least_morning_(sappada_forni_villa,5).
forecasted_rain(trieste,0,autumn) :- temperature_decreased_at_afternoon_(trieste), city_covered_less_than_(trieste,4).
forecasted_sky(pordenone,"cloudy",winter) :- city_covered_at_least_afternoon_(pordenone,4).
forecasted_sky(gorizia,"mostly_clear",autumn) :- wind_blowing_afternoon_(gorizia,"NE"), not_city_covered_at_least_morning_(gorizia,1).
forecasted_rain(sappada_forni_villa,0,autumn) :- not_city_covered_at_least_afternoon_(sappada_forni_villa,6), wind_blowing_morning_(sappada_forni_villa,"S").
forecasted_sky(gemona_stolvizza,"cloudy",spring) :- city_covered_at_least_morning_(gemona_stolvizza,3).
forecasted_rain(gemona_stolvizza,6,spring) :- city_covered_at_least_afternoon_(gemona_stolvizza,4), wind_blowing_morning_(gemona_stolvizza,"N").
forecasted_rain(barcis,0,summer) :- not_city_covered_at_least_afternoon_(barcis,2).
forecasted_sky(barcis,"mostly_clear",summer) :- not_city_covered_less_than_(barcis,1), city_covered_less_than_(barcis,5).
forecasted_rain(trieste,0,summer) :- not_city_covered_less_than_(trieste,2), not_city_covered_at_least_morning_(trieste,4).
forecasted_sky(sappada_forni_villa,"partly_cloudy",autumn) :- city_covered_less_than_(sappada_forni_villa,4), not_city_covered_less_than_(sappada_forni_villa,3), temp_front_afternoon_at_300hPa(sappada_forni_villa,udine_palmanova).
forecasted_sky(trieste,"sunny",summer) :- city_covered_at_least_morning_(trieste,2), not_city_covered_at_least_morning_(trieste,4).
forecasted_rain(sappada_forni_villa,6,spring) :- city_covered_at_least_morning_(sappada_forni_villa,5), city_covered_at_least_afternoon_(sappada_forni_villa,6).
forecasted_sky(gorizia,"sunny",autumn) :- city_covered_at_least_afternoon_(gorizia,3), temp_front_morning_at_1000hPa(gorizia,pontebba_tarvisio).
forecasted_rain(pontebba_tarvisio,4,winter) :- city_covered_exactly_afternoon_(pontebba_tarvisio,4), city_covered_at_least_morning_(pontebba_tarvisio,4).
forecasted_sky(barcis,"mostly_cloudy",autumn) :- wind_blowing_afternoon_(barcis,"N"), not_city_covered_at_least_morning_(barcis,3).
forecasted_sky(sappada_forni_villa,"mostly_cloudy",summer) :- city_covered_at_least_afternoon_(sappada_forni_villa,2), temp_front_morning_at_700hPa(sappada_forni_villa,udine_palmanova).
forecasted_sky(udine_palmanova,"mostly_cloudy",autumn) :- city_covered_at_least_morning_(udine_palmanova,2), temperature_increased_at_afternoon_(udine_palmanova).
forecasted_sky(gemona_stolvizza,"mostly_cloudy",autumn) :- wind_blowing_morning_(gemona_stolvizza,"SE"), not_city_covered_at_least_afternoon_(gemona_stolvizza,1).
forecasted_rain(gorizia,4,spring) :- city_covered_at_least_afternoon_(gorizia,6).
forecasted_sky(pordenone,"mostly_clear",spring) :- city_covered_less_than_(pordenone,1).
forecasted_rain(gorizia,2,autumn) :- city_covered_at_least_afternoon_(gorizia,2), city_covered_less_than_(gorizia,3).
forecasted_rain(pordenone,2,winter) :- city_covered_at_least_morning_(pordenone,4), wind_blowing_afternoon_(pordenone,"SE").
forecasted_rain(trieste,6,winter) :- not_city_covered_less_than_(trieste,5), city_covered_exactly_morning_(trieste,3).
forecasted_rain(pontebba_tarvisio,2,autumn) :- city_covered_at_least_morning_(pontebba_tarvisio,4), city_covered_exactly_morning_(pontebba_tarvisio,4).
forecasted_sky(trieste,"mostly_clear",autumn) :- not_city_covered_at_least_afternoon_(trieste,5), wind_blowing_afternoon_(trieste,"SE").
forecasted_sky(udine_palmanova,"mostly_clear",winter) :- city_covered_exactly_afternoon_(udine_palmanova,3), not_city_covered_at_least_morning_(udine_palmanova,2).
forecasted_rain(gemona_stolvizza,4,summer) :- city_covered_at_least_morning_(gemona_stolvizza,3), humidity_increased_at_afternoon_(gemona_stolvizza).
forecasted_rain(trieste,0,winter) :- not_city_covered_less_than_(trieste,3), wind_blowing_afternoon_(trieste,"S").
forecasted_sky(lignano_grado,"cloudy",autumn) :- city_covered_at_least_morning_(lignano_grado,5), temperature_increased_at_afternoon_(lignano_grado).
forecasted_sky(barcis,"mostly_clear",winter) :- city_covered_at_least_afternoon_(barcis,3), temp_front_morning_at_1000hPa(barcis,sappada_forni_villa).
forecasted_rain(udine_palmanova,0,winter) :- city_covered_exactly_afternoon_(udine_palmanova,3), not_city_covered_at_least_afternoon_(udine_palmanova,4).
forecasted_sky(gorizia,"sunny",summer) :- city_covered_less_than_(gorizia,1), wind_blowing_afternoon_(gorizia,"S").
forecasted_sky(gorizia,"partly_cloudy",summer) :- not_city_covered_less_than_(gorizia,6).
forecasted_sky(lignano_grado,"partly_cloudy",winter) :- wind_blowing_morning_neighbour(trieste,"N"), not_city_covered_at_least_morning_(lignano_grado,5).
forecasted_rain(gorizia,0,summer) :- not_city_covered_at_least_afternoon_(gorizia,2), wind_blowing_afternoon_(gorizia,"NE").
forecasted_rain(trieste,4,spring) :- city_covered_at_least_morning_(trieste,1), wind_blowing_morning_(trieste,"N").
forecasted_rain(barcis,0,winter) :- not_city_covered_at_least_morning_(barcis,3).
forecasted_sky(gemona_stolvizza,"mostly_clear",summer) :- humidity_decreased_at_afternoon_neighbour(pontebba_tarvisio), not_city_covered_at_least_afternoon_(gemona_stolvizza,5).
forecasted_rain(pordenone,6,spring) :- wind_blowing_afternoon_(pordenone,"N"), city_covered_at_least_afternoon_(pordenone,4).
forecasted_rain(udine_palmanova,0,autumn) :- city_covered_less_than_(udine_palmanova,2).
forecasted_sky(sappada_forni_villa,"mostly_cloudy",autumn) :- city_covered_at_least_morning_(sappada_forni_villa,4).
forecasted_sky(pontebba_tarvisio,"partly_cloudy",winter) :- temperature_decreased_at_afternoon_(pontebba_tarvisio), humidity_decreased_at_afternoon_(pontebba_tarvisio), not_city_covered_at_least_morning_(pontebba_tarvisio,4).
forecasted_sky(sappada_forni_villa,"mostly_clear",winter) :- not_city_covered_at_least_afternoon_(sappada_forni_villa,2), wind_blowing_afternoon_neighbour(gemona_stolvizza,"SE").
forecasted_rain(gorizia,0,winter) :- city_covered_at_least_morning_(gorizia,1), not_city_covered_at_least_morning_(gorizia,3).
forecasted_rain(lignano_grado,6,autumn) :- city_covered_at_least_morning_(lignano_grado,6).
forecasted_rain(udine_palmanova,0,autumn) :- not_city_covered_at_least_morning_(udine_palmanova,3), hum_front_afternoon_at_1000hPa(udine_palmanova,gemona_stolvizza).
forecasted_sky(gemona_stolvizza,"mostly_clear",autumn) :- not_city_covered_at_least_morning_(gemona_stolvizza,3), humidity_increased_at_afternoon_(gemona_stolvizza).
forecasted_sky(pontebba_tarvisio,"partly_cloudy",spring) :- city_covered_at_least_morning_(pontebba_tarvisio,3), not_city_covered_at_least_afternoon_(pontebba_tarvisio,3).
forecasted_sky(pordenone,"mostly_cloudy",autumn) :- temperature_increased_at_afternoon_neighbour(udine_palmanova), city_covered_at_least_morning_(pordenone,2).
forecasted_rain(udine_palmanova,1,winter) :- city_covered_at_least_morning_(udine_palmanova,3), not_city_covered_at_least_afternoon_(udine_palmanova,2).
forecasted_sky(pordenone,"partly_cloudy",winter) :- city_covered_less_than_(pordenone,1).
forecasted_sky(pontebba_tarvisio,"mostly_clear",winter) :- temperature_increased_at_afternoon_(pontebba_tarvisio), not_city_covered_at_least_morning_(pontebba_tarvisio,1).
forecasted_sky(lignano_grado,"mostly_cloudy",winter) :- city_covered_at_least_morning_(lignano_grado,3), city_covered_exactly_morning_(lignano_grado,3).
forecasted_rain(gorizia,2,autumn) :- city_covered_at_least_morning_(gorizia,2), not_city_covered_at_least_afternoon_(gorizia,1).
forecasted_sky(gemona_stolvizza,"mostly_cloudy",summer) :- city_covered_at_least_morning_(gemona_stolvizza,3), humidity_increased_at_afternoon_(gemona_stolvizza).
forecasted_rain(barcis,1,autumn) :- city_covered_exactly_morning_(barcis,3), not_city_covered_at_least_afternoon_(barcis,1).
forecasted_rain(trieste,2,autumn) :- city_covered_at_least_afternoon_(trieste,3), wind_blowing_afternoon_(trieste,"NE").
forecasted_sky(pontebba_tarvisio,"mostly_cloudy",autumn) :- not_city_covered_less_than_(pontebba_tarvisio,6), city_covered_at_least_morning_(pontebba_tarvisio,4).
forecasted_rain(gemona_stolvizza,2,spring) :- wind_blowing_afternoon_(gemona_stolvizza,"N"), not_city_covered_at_least_morning_(gemona_stolvizza,6).
forecasted_rain(gemona_stolvizza,1,winter) :- wind_blowing_morning_(gemona_stolvizza,"NE"), not_city_covered_at_least_afternoon_(gemona_stolvizza,2).
forecasted_sky(pontebba_tarvisio,"cloudy",spring) :- city_covered_at_least_afternoon_(pontebba_tarvisio,3).
forecasted_rain(lignano_grado,0,autumn) :- city_covered_exactly_afternoon_(lignano_grado,2), not_city_covered_at_least_afternoon_(lignano_grado,3).
forecasted_rain(gemona_stolvizza,2,autumn) :- city_covered_at_least_morning_(gemona_stolvizza,3), not_city_covered_less_than_(gemona_stolvizza,5), city_covered_less_than_(gemona_stolvizza,6).
forecasted_sky(gorizia,"sunny",summer) :- wind_blowing_morning_(gorizia,"SE"), city_covered_less_than_(gorizia,2).
forecasted_sky(pontebba_tarvisio,"mostly_clear",autumn) :- city_covered_at_least_afternoon_(pontebba_tarvisio,4), wind_blowing_morning_(pontebba_tarvisio,"SE").
forecasted_sky(barcis,"mostly_cloudy",autumn) :- city_covered_at_least_afternoon_(barcis,3).
forecasted_rain(gorizia,4,autumn) :- wind_blowing_afternoon_(gorizia,"N"), not_city_covered_at_least_afternoon_(gorizia,2).
forecasted_sky(gemona_stolvizza,"mostly_clear",autumn) :- city_covered_at_least_afternoon_(gemona_stolvizza,3), wind_blowing_afternoon_(gemona_stolvizza,"SE").
forecasted_sky(udine_palmanova,"partly_cloudy",spring) :- temperature_decreased_at_afternoon_neighbour(sappada_forni_villa), city_covered_less_than_(udine_palmanova,5).
forecasted_sky(trieste,"partly_cloudy",autumn) :- humidity_increased_at_afternoon_(trieste), not_city_covered_at_least_morning_(trieste,3), wind_blowing_morning_neighbour(lignano_grado,"E").
forecasted_sky(trieste,"partly_cloudy",winter) :- not_city_covered_less_than_(trieste,3), temperature_increased_at_afternoon_neighbour(lignano_grado), not_city_covered_at_least_afternoon_(trieste,4).
forecasted_rain(pordenone,2,autumn) :- not_city_covered_at_least_afternoon_(pordenone,1), temp_front_afternoon_at_1000hPa(pordenone,udine_palmanova).
forecasted_rain(gemona_stolvizza,4,spring) :- wind_blowing_afternoon_(gemona_stolvizza,"E"), city_covered_at_least_morning_(gemona_stolvizza,5).
forecasted_sky(udine_palmanova,"cloudy",autumn) :- city_covered_at_least_afternoon_(udine_palmanova,3), not_city_covered_less_than_(udine_palmanova,6).
forecasted_rain(udine_palmanova,4,autumn) :- city_covered_at_least_afternoon_(udine_palmanova,5).
forecasted_sky(pordenone,"mostly_clear",autumn) :- temperature_decreased_at_afternoon_neighbour(lignano_grado), hum_front_afternoon_at_1000hPa(pordenone,barcis), not_city_covered_at_least_morning_(pordenone,3).
forecasted_rain(barcis,4,autumn) :- city_covered_at_least_afternoon_(barcis,4).
forecasted_sky(pordenone,"sunny",summer) :- not_city_covered_at_least_morning_(pordenone,1).
forecasted_sky(pontebba_tarvisio,"mostly_cloudy",winter) :- city_covered_exactly_afternoon_(pontebba_tarvisio,1), city_covered_at_least_morning_(pontebba_tarvisio,4).
forecasted_rain(barcis,6,spring) :- city_covered_at_least_morning_(barcis,5), city_covered_at_least_afternoon_(barcis,5).
forecasted_sky(lignano_grado,"cloudy",autumn) :- city_covered_at_least_afternoon_(lignano_grado,4).
forecasted_rain(trieste,2,spring) :- city_covered_at_least_afternoon_(trieste,4).
forecasted_sky(pontebba_tarvisio,"cloudy",winter) :- city_covered_at_least_morning_(pontebba_tarvisio,5).
forecasted_sky(barcis,"mostly_cloudy",spring) :- city_covered_at_least_afternoon_(barcis,1).