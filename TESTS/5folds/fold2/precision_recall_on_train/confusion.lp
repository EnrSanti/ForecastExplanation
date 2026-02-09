% True Positive: predico correttamente (anche se ho altre predizioni sbagliate)

tp(W, S) :- forecasted_sky(L, W, S), gt_forecasted_sky(L, W, S), location_considered(L), possible_season(S).
tp(R, S) :- forecasted_rain(L, R, S), gt_forecasted_rain(L, R, S), location_considered(L), possible_season(S), lv(R).


% False positive: sto predicendo la mia classe piuttosto che unaltra

fp(W1, S) :- forecasted_sky(L, W1, S), gt_forecasted_sky(L, W2, S), W1!=W2, location_considered(L), possible_season(S).
fp(W, S1) :- forecasted_sky(L, W, S1), gt_forecasted_sky(L, W, S2), S1!=S2, location_considered(L), possible_season(S).
fp(W1, S1) :- forecasted_sky(L, W1, S1), gt_forecasted_sky(L, W2, S2), S1!=S2, W1!=W2, location_considered(L), possible_season(S).

fp(R1, S) :- forecasted_rain(L, R1, S), gt_forecasted_rain(L, R2, S), R1!=R2, location_considered(L), possible_season(S), lv(R1), lv(R2).
fp(R, S1) :- forecasted_rain(L, R, S1), gt_forecasted_rain(L, R, S2), S1!=S2, location_considered(L), possible_season(S), lv(R).
fp(R1, S1) :- forecasted_rain(L, R1, S1), gt_forecasted_rain(L, R2, S2), S1!=S2, R1!=R2, location_considered(L), possible_season(S), lv(R1), lv(R2).

% False negative: predico solo cose sbagliate

fn(W1, S) :- forecasted_sky(L, W1, S), gt_forecasted_sky(L, W2, S), W1!=W2, location_considered(L), possible_season(S), not tp(W1, S).
fn(R1, S) :- forecasted_rain(L, R1, S), gt_forecasted_rain(L, R2, S), R1!=R2, location_considered(L), possible_season(S), not tp(L, R1).

% True negative: predico correttamente, ma non la classe di interesse

tn(W1, S1):- forecasted_sky(L, W, S), gt_forecasted_sky(L, W, S), W!=W1, location_considered(L), possible_season(S), possible_season(S1), coverage(W), coverage(W1).
tn(R1, S1):- forecasted_rain(L, R, S), gt_forecasted_rain(L, R, S), S!=S1, location_considered(L), possible_season(S), possible_season(S1), lv(R1), lv(R).


#show tp/2.
#show tn/2.
#show fp/2.
#show fn/2.