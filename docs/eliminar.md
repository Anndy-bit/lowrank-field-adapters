Resumen de scripts:
Comando	Qué hace
make venv	Solo primera vez - instala dependencias
make svd	Descarga modelo + calcula SVD factors (tolerante a cortes)
make train	Entrena + monitoring + benchmarks (USA esto)
make evaluate	Solo corre benchmarks sobre checkpoint existente
make test	Tests unitarios