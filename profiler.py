"""Cuestionario de perfil de riesgo. 6 preguntas -> perfil + horizonte."""
QUESTIONS = [
    {"id":"horizon","text":"Cuando previsiblemente necesitaras este dinero?",
     "options":[("Menos de 3 anos",0),("Entre 3 y 7 anos",1),
                ("Entre 7 y 15 anos",2),("Mas de 15 anos",3)],"weight":2.0},
    {"id":"reaction","text":"Si tu cartera cae un 25% en 3 meses, que harias?",
     "options":[("Vender todo y poner en cuenta",-2),
                ("Vender una parte para reducir riesgo",-1),
                ("Mantener y esperar",1),
                ("Comprar mas aprovechando precios bajos",2)],"weight":2.0},
    {"id":"experience","text":"Que experiencia tienes invirtiendo?",
     "options":[("Ninguna",0),("Algun deposito o fondo basico",1),
                ("ETFs / acciones desde hace anos",2),
                ("Activa: opciones, derivados, etc.",3)],"weight":1.0},
    {"id":"income_dep","text":"Que % de tus ingresos depende de mantener este dinero a salvo?",
     "options":[("Es mi unico ahorro",-2),("Una parte significativa",-1),
                ("Es prescindible",1),("Es solo para invertir y crecer",2)],"weight":1.5},
    {"id":"return_pref","text":"Que prefieres a largo plazo?",
     "options":[("Rentabilidad baja pero estable (3-5%)",0),
                ("Rentabilidad media con caidas moderadas (6-8%)",1),
                ("Rentabilidad alta con caidas fuertes ocasionales (9-11%)",2),
                ("Maxima rentabilidad posible aunque haya caidas grandes (>11%)",3)],
     "weight":1.5},
    {"id":"loss_tolerance","text":"Cual es la maxima caida temporal que aceptarias?",
     "options":[("Hasta -10%",0),("Hasta -20%",1),("Hasta -35%",2),
                ("Mas de -40% si la rentabilidad esperada lo justifica",3)],"weight":2.0},
]

def score_to_profile(total_score):
    """Mapea score total a perfil."""
    if total_score < 5:    return "Conservador"
    if total_score < 12:   return "Moderado"
    if total_score < 18:   return "Crecimiento"
    return "Agresivo"

def horizon_from_answer(answer_idx):
    return ["Corto (<3 anos)","Medio (3-7 anos)",
            "Largo (7-15 anos)","Muy largo (>15 anos)"][answer_idx]

def evaluate(answers):
    """answers: dict question_id -> indice de opcion seleccionada (0..n-1)."""
    score = 0
    horizon = "Medio (3-7 anos)"
    for q in QUESTIONS:
        if q["id"] not in answers: continue
        idx = answers[q["id"]]
        if idx >= len(q["options"]): continue
        val = q["options"][idx][1]
        score += val * q["weight"]
        if q["id"] == "horizon":
            horizon = horizon_from_answer(idx)
    profile = score_to_profile(score)
    return {"profile": profile, "horizon": horizon, "score": score}
