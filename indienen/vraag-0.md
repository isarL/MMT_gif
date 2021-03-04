Door met numpy-rijen te werken, kunnen deze in 1 bewerking van elkaar afgetrokken worden. Dit resultaat is opnieuw een numpy-rij, waardoor deze ook in 1 keer gekwadrateerd kan worden en er vervolgens broadcasting kan plaatsvinden van de factor 1/(m*n*3). Zo wordt de MSE berekend. Vervolgens kan de log van een numpy-rij genomen worden en weer een broadcast van de factor 10 plaatsvinden. Zo wordt de PSNR bepaald. Ook gebeurt er een implicit cast van de MSE naar een undefined long long, om zo komma-getallen te behouden, maar zo dat het ook zeker groot genoeg is, zelfs voor afbeeldingen die sterk verschillen. Dit is om overflow te voorkomen.

Om hier dus een vlotte werking te verkijgen is het gebruik van numpy-rijen, met mogelijke broadcasts en de toepassingen ervan.