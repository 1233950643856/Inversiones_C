# Inversiones PRO

Sistema profesional de soporte a la decision de inversion (DSS) optimizado para
presupuestos pequenos (300-700 EUR). Metodologia institucional: Ledoit-Wolf,
Black-Litterman, HRP, Min-CVaR, walk-forward ML, stress test historico.

**Aviso:** herramienta exclusivamente informativa. NO es asesoramiento financiero.

## Despliegue en Streamlit Community Cloud

1. Sube esta carpeta a un repositorio de GitHub (publico o privado).
2. Entra en https://share.streamlit.io con tu cuenta de GitHub.
3. Pulsa "New app" y selecciona el repo + rama main + archivo `app.py`.
4. La primera build tarda 3-5 minutos.
5. Te dara una URL publica del tipo `https://tu-app.streamlit.app`.

## Instalar como PWA en el movil

### iPhone / iPad (Safari)
1. Abre la URL publica de la app en Safari.
2. Pulsa el boton "Compartir" (el cuadradito con flecha hacia arriba).
3. Selecciona "Anadir a la pantalla de inicio".
4. Confirma. Aparecera un icono "InvPRO" en tu home como cualquier otra app.

### Android (Chrome)
1. Abre la URL publica de la app en Chrome.
2. Pulsa el menu de los tres puntos arriba a la derecha.
3. Selecciona "Anadir a la pantalla de inicio" (o "Instalar app" si Chrome detecta el manifest).
4. Confirma.

Una vez instalada, la app se abre a pantalla completa, sin barra del navegador,
como cualquier otra app del telefono.

## Estructura del proyecto

- `app.py` - punto de entrada Streamlit (10 paginas)
- `config.py` - universo de 34 activos + brokers + perfiles
- `data_loader.py` - descarga Yahoo Finance + cache
- `feature_engineering.py` - features tecnicos
- `ml_predictor.py` - ensemble XGB + RF + Ridge
- `optimizer.py` - Markowitz LW + Min-CVaR + HRP + Black-Litterman + balanced + income + all-weather
- `backtester.py` - rebalanceo por umbral + slippage + fiscalidad
- `allocator.py` - presupuesto -> operaciones concretas
- `metrics.py` - Sharpe, Sortino, Calmar, CVaR, Omega, Ulcer, etc.
- `profiler.py` - cuestionario de perfil de riesgo
- `scheduler.py` - APScheduler para refresco diario
- `requirements.txt` - dependencias para Streamlit Cloud
- `manifest.json` - manifiesto PWA
- `.streamlit/config.toml` - tema oscuro + ajustes servidor
