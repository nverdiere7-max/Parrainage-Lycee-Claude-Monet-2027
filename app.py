"""
Emballement thermique dans un réacteur fermé
Hydrolyse de l'anhydride acétique : (CH3CO)2O + H2O -> 2 CH3COOH

Application Streamlit — projet lycée (terminales spé maths / maths expertes)
Lancement local :  streamlit run app.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Constantes physiques (littérature)
# ---------------------------------------------------------------------------
R = 8.314            # J/(mol.K)
K0 = 1.6e7           # facteur pré-exponentiel (min^-1)  — Asiedu et al.
EA = 50.5e3          # énergie d'activation (J/mol)       — Asiedu et al.
DH = -53.4e3         # enthalpie de réaction (J/mol)      — Asiedu et al.
CP_ANHYDRIDE = 189.7   # capacité calorifique molaire de l'anhydride (J/(mol.K))  — Asiedu et al.
CP_EAU = 75.4          # capacité calorifique molaire de l'eau (J/(mol.K))        — Asiedu et al.
M_ANHYDRIDE = 102.09   # masse molaire de l'anhydride (g/mol)
M_EAU = 18.02          # masse molaire de l'eau (g/mol)
RHO_ANHYDRIDE = 1.08   # masse volumique de l'anhydride (g/mL)
RHO_EAU = 1.00         # masse volumique de l'eau (g/mL)

# Valeurs nominales : 1 mol d'anhydride pour 10 mol d'eau, environ 2 L de mélange
NOMINAL = dict(
    n0=7.3,          # quantité initiale d'anhydride (mol)
    n_eau=73.0,      # quantité d'eau (mol)
    T0=25.0,         # température initiale (°C)
    Te=25.0,         # température du fluide de refroidissement (°C)
    UA=10.0,         # coefficient d'échange (W/K)
    t_panne=None,    # instant de la panne (min)
    perte=0.0,       # perte d'efficacité du refroidissement (0 à 1)
)


def volume_L(n0, n_eau):
    """Volume du mélange chargé (L), en négligeant la contraction au mélange."""
    return (n0 * M_ANHYDRIDE / RHO_ANHYDRIDE + n_eau * M_EAU / RHO_EAU) / 1000


def capacite_thermique(n0, n_eau):
    """Capacité thermique totale du contenu du réacteur (J/K)."""
    return n0 * CP_ANHYDRIDE + n_eau * CP_EAU


DUREE = 60.0         # durée simulée (min)
TAU_PANNE = 1.0      # durée de la chute de UA lors d'une panne (min)


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
def k(T_kelvin):
    """Constante de vitesse (min^-1), loi d'Arrhenius."""
    return K0 * np.exp(-EA / (R * T_kelvin))


def ua_effectif(t, p):
    """Coefficient d'échange au temps t (W/K), avec panne éventuelle."""
    if p["t_panne"] is None or t < p["t_panne"]:
        return p["UA"]
    return p["UA"] * (1 - p["perte"] * (1 - np.exp(-(t - p["t_panne"]) / TAU_PANNE)))


def second_membre(t, y, p):
    n, T = y                          # n : quantité d'anhydride restante (mol), T en K
    Te = p["Te"] + 273.15
    Ctot = capacite_thermique(p["n0"], p["n_eau"])           # J/K
    vitesse = k(T) * n                                       # mol/min
    dn = -vitesse
    chaleur_produite = (-DH) * vitesse / Ctot                # K/min
    chaleur_evacuee = ua_effectif(t, p) * 60 * (T - Te) / Ctot   # K/min
    return [dn, chaleur_produite - chaleur_evacuee]


@st.cache_data
def simuler(n0, n_eau, T0, Te, UA, t_panne, perte):
    p = dict(n0=n0, n_eau=n_eau, T0=T0, Te=Te, UA=UA, t_panne=t_panne, perte=perte)
    t = np.linspace(0, DUREE, 3001)
    sol = solve_ivp(second_membre, (0, DUREE), [n0, T0 + 273.15],
                    t_eval=t, args=(p,), max_step=0.05, rtol=1e-8, atol=1e-8)
    n = sol.y[0]
    T = sol.y[1] - 273.15
    dT = np.array([second_membre(ti, yi, p)[1] for ti, yi in zip(sol.t, sol.y.T)])
    d2T = np.gradient(dT, t)
    return t, n, T, dT, d2T


DUREE_MIN_ZONE = 1.0  # une zone d'emballement doit durer au moins 1 min


def zone_emballement(dT, d2T, t=None):
    """Instants où dT/dt > 0 ET d²T/dt² > 0.
    Les zones de moins de DUREE_MIN_ZONE minute sont ignorées
    (simples « à-coups » de la courbe, pas un vrai emballement)."""
    brut = (dT > 1e-4) & (d2T > 1e-5)
    if t is None:
        t = np.linspace(0, DUREE, len(dT))
    zone = brut.copy()
    i = 0
    while i < len(brut):
        if brut[i]:
            j = i
            while j < len(brut) and brut[j]:
                j += 1
            if t[j - 1] - t[i] < DUREE_MIN_ZONE:
                zone[i:j] = False
            i = j
        else:
            i += 1
    return zone


def premier_instant(t, masque):
    return float(t[np.argmax(masque)]) if masque.any() else None


# ---------------------------------------------------------------------------
# Scénarios de faute
# ---------------------------------------------------------------------------
SCENARIOS = {
    "Fonctionnement nominal": None,
    "Surcharge en anhydride": dict(
        cle="n0", label="Surcharge par rapport au chargement nominal (%)",
        mini=0, maxi=100, defaut=60, pas=5,
        vers_param=lambda x: {"n0": NOMINAL["n0"] * (1 + x / 100)},
        explication="L'opérateur a versé trop d'anhydride acétique. Une surcharge de x % signifie "
                    "qu'on a versé x % d'anhydride en plus de la quantité prévue (7,3 mol, soit environ "
                    "745 g), sans changer la quantité d'eau. Il y a plus de réactif, donc plus de "
                    "chaleur à évacuer."),
    "Panne du refroidissement": dict(
        cle="perte", label="Perte d'efficacité du refroidissement (%)",
        mini=0, maxi=100, defaut=100, pas=5,
        vers_param=lambda x: {"perte": x / 100},
        explication="La pompe du fluide de refroidissement ralentit ou s'arrête. Une perte de x % "
                    "signifie que l'enveloppe n'évacue plus que (100 − x) % de la chaleur qu'elle "
                    "évacuerait normalement à la même température : UA est multiplié par (1 − x/100). "
                    "L'enveloppe n'évacue plus (ou presque plus) la chaleur."),
    "Fluide de refroidissement trop chaud": dict(
        cle="Te", label="Température du fluide de refroidissement (°C)",
        mini=15, maxi=60, defaut=40, pas=1,
        vers_param=lambda x: {"Te": float(x)},
        explication="Le fluide qui circule dans l'enveloppe est plus chaud que prévu "
                    "(groupe froid défaillant, été caniculaire...)."),
    "Préchauffage excessif": dict(
        cle="T0", label="Température initiale du mélange (°C)",
        mini=15, maxi=60, defaut=50, pas=1,
        vers_param=lambda x: {"T0": float(x)},
        explication="Le mélange a été chargé trop chaud. La réaction démarre "
                    "d'emblée très vite."),
}


def fr(x, n=1):
    """Nombre écrit à la française (virgule décimale)."""
    return f"{x:.{n}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Emballement thermique", layout="wide")
st.title("Emballement thermique dans un réacteur chimique")

SEUIL_DANGER = 60  # limite de sécurité (°C) utilisée pour classer les essais de l'enquête

onglet_exp, onglet_enquete, onglet_simu, onglet_seuil = st.tabs(
    ["L'expérience", "Enquête : reconnaître un emballement",
     "Simulation d'un scénario", "À partir de quand y a-t-il un problème ?"])

# ----------------------------------------------------------------- Onglet 1
with onglet_exp:
    st.header("Le dispositif")
    st.markdown(f"""
On étudie un réacteur fermé à double enveloppe de 3 L : une cuve agitée,
entourée d'une enveloppe dans laquelle circule un fluide de refroidissement.

On y verse {fr(NOMINAL['n0'])} mol d'anhydride acétique (environ {NOMINAL['n0'] * M_ANHYDRIDE:.0f} g)
et {fr(NOMINAL['n_eau'], 0)} mol d'eau (environ {NOMINAL['n_eau'] * M_EAU:.0f} g), soit 1 mole d'anhydride
pour 10 moles d'eau et environ {fr(volume_L(NOMINAL['n0'], NOMINAL['n_eau']))} L de mélange.
Il se produit la réaction suivante, qui dégage de la chaleur :
""")
    st.latex(r"(\mathrm{CH_3CO})_2\mathrm{O} + \mathrm{H_2O} \longrightarrow 2\,\mathrm{CH_3COOH}")
    st.markdown("Une sonde mesure la température du mélange. C'est la seule grandeur mesurée.")

    st.header("Le modèle mathématique")
    st.markdown("On note $n(t)$ la quantité d'anhydride restante (en mol) et $T(t)$ la température du mélange.")
    st.markdown("1. Le réactif est consommé (l'eau est en large excès) :")
    st.latex(r"\frac{dn}{dt} = -k(T)\,n \qquad \text{avec} \qquad "
             r"k(T) = k_0\, e^{-E_a/(R\,T)} \quad \text{(loi d'Arrhenius)}")
    st.markdown("2. Bilan thermique :")
    st.latex(r"C_{\text{th}}\,\frac{dT}{dt} = \underbrace{(-\Delta H)\,k(T)\,n}_{\text{chaleur produite}}"
             r"\;-\;\underbrace{UA\,(T - T_e)}_{\text{chaleur évacuée}}")
    st.markdown("où $C_{\\text{th}}$ est la capacité thermique du contenu du réacteur : l'énergie qu'il "
                "faut lui fournir pour l'échauffer de 1 °C. Elle dépend de ce qu'on a versé :")
    st.latex(r"C_{\text{th}} = n_0\, c_{\text{anhydride}} + n_{\text{eau}}\, c_{\text{eau}}")
    st.markdown("""
Le cercle vicieux de l'emballement : plus il fait chaud, plus $k(T)$ est grand, donc plus la réaction
va vite, donc plus elle produit de chaleur... et plus il fait chaud. L'emballement survient quand la chaleur
produite l'emporte durablement sur la chaleur évacuée.
""")

    st.header("Les paramètres")
    st.markdown(f"""
| Paramètre | Signification | Valeur nominale |
|---|---|---|
| $k_0$ | facteur pré-exponentiel | {K0:.1e} min⁻¹ |
| $E_a$ | énergie d'activation | {EA/1000:.1f} kJ/mol |
| $\\Delta H$ | enthalpie de réaction | {DH/1000:.1f} kJ/mol |
| $c_{{\\text{{anhydride}}}}$ | capacité thermique molaire de l'anhydride | {CP_ANHYDRIDE} J/(mol·K) |
| $c_{{\\text{{eau}}}}$ | capacité thermique molaire de l'eau | {CP_EAU} J/(mol·K) |
| $n_0$ | quantité initiale d'anhydride | {fr(NOMINAL['n0'])} mol |
| $n_{{\\text{{eau}}}}$ | quantité d'eau | {fr(NOMINAL['n_eau'], 0)} mol |
| $C_{{\\text{{th}}}}$ | capacité thermique du contenu | {capacite_thermique(NOMINAL['n0'], NOMINAL['n_eau']):.0f} J/K |
| $T_0$ | température initiale | {NOMINAL['T0']:.0f} °C |
| $T_e$ | température du fluide de refroidissement | {NOMINAL['Te']:.0f} °C |
| $UA$ | efficacité de l'échange thermique | {NOMINAL['UA']:.0f} W/K |

*Valeurs cinétiques : Asiedu, Hildebrandt et Glasser (hydrolyse de l'anhydride acétique en vase Dewar).
Enthalpie cohérente avec le NIST WebBook.*
""")

    st.header("Les fautes possibles")
    for nom, sc in SCENARIOS.items():
        if sc is not None:
            st.markdown(f"- {nom} : {sc['explication']}")

    st.header("Quand faut-il donner l'alerte ?")
    st.markdown("""
Une première idée : donner l'alerte quand la température dépasse une limite de sécurité.
C'est simple... mais à ce moment-là, le danger est déjà là.

Votre mission : trouver un critère qui permette de reconnaître un emballement plus tôt,
en observant les courbes de l'onglet *Enquête*.
""")


# ----------------------------------------------------------------- Enquête
ESSAIS = [  # (paramètres modifiés, cause) — l'ordre est volontairement mélangé
    ({"n0": NOMINAL["n0"] * 1.6}, "surcharge de 60 %"),
    ({}, "fonctionnement nominal"),
    ({"Te": 45.0}, "fluide de refroidissement à 45 °C"),
    ({"n0": NOMINAL["n0"] * 1.1}, "surcharge de 10 %"),
    ({"t_panne": 8.0, "perte": 1.0}, "panne totale du refroidissement à t = 8 min"),
    ({"Te": 30.0}, "fluide de refroidissement à 30 °C"),
    ({"T0": 50.0}, "mélange chargé à 50 °C"),
    ({"T0": 32.0}, "mélange chargé à 32 °C"),
]
GRANDEURS = {
    "Température T (°C)": 2,
    "Dérivée première dT/dt (°C/min)": 3,
    "Dérivée seconde d²T/dt² (°C/min²)": 4,
}

with onglet_enquete:
    st.markdown(f"""
Voici huit essais réalisés dans le réacteur. Pour chacun, on sait s'il y a eu emballement
(la température a dépassé la limite de sécurité de {SEUIL_DANGER} °C) ou pas d'emballement.

Question : qu'ont en commun les courbes des essais avec emballement, *bien avant*
que la température ne dépasse {SEUIL_DANGER} °C ? Et qu'est-ce qui les distingue des autres ?
""")
    c1, c2, c3 = st.columns([2, 1, 1])
    grandeur = c1.radio("Courbe à afficher", list(GRANDEURS.keys()), horizontal=True)
    vue = c2.radio("Présentation", ["Un graphique par essai", "Tout superposer"])
    t_fin = c3.slider("Afficher jusqu'à t = (min)", 5, 60, 30, 5)
    montrer_cause = st.checkbox("Révéler la cause de chaque essai", value=False, key="cause")

    resultats = []
    for num, (modif, cause) in enumerate(ESSAIS, 1):
        q = dict(NOMINAL); q.update(modif)
        res = simuler(q["n0"], q["n_eau"], q["T0"], q["Te"], q["UA"], q["t_panne"], q["perte"])
        emb = res[2].max() > SEUIL_DANGER
        resultats.append((num, cause, emb, res))

    idx = GRANDEURS[grandeur]
    ROUGE, VERT = "#c0392b", "#27ae60"

    if vue == "Tout superposer":
        fig = go.Figure()
        for num, cause, emb, res in resultats:
            t, y = res[0], res[idx]
            m = t <= t_fin
            nom_courbe = f"Essai {num}" + (f" ({cause})" if montrer_cause else "")
            fig.add_trace(go.Scatter(x=t[m], y=y[m], mode="lines", name=nom_courbe,
                                     line=dict(color=ROUGE if emb else VERT, width=2)))
        if idx > 2:
            fig.add_hline(y=0, line=dict(color="grey", width=1))
        else:
            fig.add_hline(y=SEUIL_DANGER, line=dict(color="orange", dash="dash"),
                          annotation_text=f"limite {SEUIL_DANGER} °C")
        fig.update_layout(height=550, xaxis_title="temps (min)", yaxis_title=grandeur,
                          title="En rouge : emballement — en vert : pas d'emballement")
        st.plotly_chart(fig, width="stretch")
    else:
        titres = []
        for num, cause, emb, _ in resultats:
            etat = "EMBALLEMENT" if emb else "pas d'emballement"
            titres.append(f"Essai {num} : {etat}" + (f"<br><sub>{cause}</sub>" if montrer_cause else ""))
        fig = make_subplots(rows=2, cols=4, subplot_titles=titres,
                            vertical_spacing=0.18, horizontal_spacing=0.06)
        for k_, (num, cause, emb, res) in enumerate(resultats):
            ligne, col = k_ // 4 + 1, k_ % 4 + 1
            t, y = res[0], res[idx]
            m = t <= t_fin
            fig.add_trace(go.Scatter(x=t[m], y=y[m], mode="lines", showlegend=False,
                                     line=dict(color=ROUGE if emb else VERT, width=2.5)),
                          row=ligne, col=col)
            if idx > 2:
                fig.add_hline(y=0, line=dict(color="grey", width=1), row=ligne, col=col)
            else:
                fig.add_hline(y=SEUIL_DANGER, line=dict(color="orange", dash="dash"),
                              row=ligne, col=col)
        fig.update_xaxes(title_text="t (min)", row=2)
        fig.update_layout(height=650, margin=dict(t=70))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Pistes pour avancer"):
        st.markdown("""
1. Sur les courbes de température, regardez le *début* de chaque essai : comment la courbe monte-t-elle ?
2. Affichez la dérivée première. Que représente-t-elle ? Quel est son signe au début ?
3. Affichez la dérivée seconde. Que dit son signe sur la forme de la courbe de température ?
4. Proposez une règle qui donne l'alerte avant que la limite de sécurité ne soit dépassée.
   Vérifiez-la sur les huit essais, puis testez-la dans l'onglet *Simulation d'un scénario*.
""")


# ----------------------------------------------------------------- Onglet 2
with onglet_simu:
    col_reglages, col_courbes = st.columns([1, 3])

    with col_reglages:
        nom = st.selectbox("Scénario", list(SCENARIOS.keys()))
        sc = SCENARIOS[nom]
        p = dict(NOMINAL)

        if sc is not None:
            st.caption(sc["explication"])
            x = st.slider(sc["label"], sc["mini"], sc["maxi"], sc["defaut"], sc["pas"])
            p.update(sc["vers_param"](x))
            if nom == "Panne du refroidissement":
                p["t_panne"] = float(st.slider("Instant de la panne (min)", 0, 40, 10, 1))

        T_seuil = st.slider("Seuil de température admissible (°C)", 30, 90, 60, 1)
        AIDE = st.checkbox("Aide : afficher les zones d'emballement", value=False, key="aide")

        with st.expander("Autres paramètres (valeurs nominales)"):
            p["UA"] = st.slider("UA (W/K)", 2.0, 30.0, float(p["UA"]), 0.5)

    t, n, T, dT, d2T = simuler(p["n0"], p["n_eau"], p["T0"], p["Te"], p["UA"],
                               p["t_panne"], p["perte"])
    zone = zone_emballement(dT, d2T, t)
    t_deriv = premier_instant(t, zone)
    t_seuil = premier_instant(t, T > T_seuil)

    with col_courbes:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Température maximale", f"{T.max():.1f} °C")
        m3.metric("Alerte « seuil »", "aucune" if t_seuil is None else f"t = {t_seuil:.1f} min")
        if AIDE:
            m2.metric("Début de la zone rouge", "aucune" if t_deriv is None else f"t = {t_deriv:.1f} min")
            if t_deriv is not None and t_seuil is not None:
                m4.metric("Temps gagné sur le seuil", f"{t_seuil - t_deriv:.1f} min",
                          help="Durée entre le début de la zone rouge et le dépassement du seuil.")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=("Température T (°C)",
                                            "Dérivée première dT/dt (°C/min)",
                                            "Dérivée seconde d²T/dt² (°C/min²)"))
        for i, (y, coul) in enumerate([(T, "#c0392b"), (dT, "#2c3e50"), (d2T, "#2c3e50")], 1):
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color=coul, width=2.5),
                                     showlegend=False), row=i, col=1)
            if i > 1:
                fig.add_hline(y=0, line=dict(color="grey", width=1), row=i, col=1)

        # zones d'emballement (dT>0 et d2T>0), sur les trois graphes
        debuts = np.where(np.diff(zone.astype(int)) == 1)[0] + 1
        fins = np.where(np.diff(zone.astype(int)) == -1)[0] + 1
        if zone[0]:
            debuts = np.r_[0, debuts]
        if zone[-1]:
            fins = np.r_[fins, len(t) - 1]
        for a, b in zip(debuts, fins) if AIDE else []:
            for i in (1, 2, 3):
                fig.add_vrect(x0=t[a], x1=t[b], fillcolor="red", opacity=0.15,
                              line_width=0, row=i, col=1)

        fig.add_hline(y=T_seuil, line=dict(color="orange", dash="dash"),
                      annotation_text=f"seuil {T_seuil} °C", row=1, col=1)
        if p["t_panne"] is not None and p["perte"] > 0:
            fig.add_vline(x=p["t_panne"], line=dict(color="blue", dash="dot"),
                          annotation_text="panne")

        fig.update_xaxes(title_text="temps (min)", row=3, col=1)
        fig.update_layout(height=750, margin=dict(t=40, b=40))
        st.plotly_chart(fig, width="stretch")
        if AIDE:
            st.caption("Zones rouges : zones d'emballement. Que se passe-t-il sur les courbes dans ces zones ? "
                       "Ligne orange : seuil de température admissible.")
        else:
            st.caption("Ligne orange : seuil de température admissible.")

# ----------------------------------------------------------------- Onglet 3
VALEUR_NOMINALE = {
    "Surcharge en anhydride": 0,
    "Panne du refroidissement": 0,
    "Fluide de refroidissement trop chaud": NOMINAL["Te"],
    "Préchauffage excessif": NOMINAL["T0"],
}

with onglet_seuil:
    st.markdown("""
Dans l'onglet *Simulation*, on regarde un scénario à la fois. Ici, on fait varier la faute
du plus anodin au plus grave, et l'on cherche à partir de quelle valeur le réacteur devient dangereux.

On dit qu'il y a dépassement quand la température du mélange franchit, à un moment quelconque des
60 minutes simulées, le seuil de température admissible choisi ci-dessous. Concrètement, on compare la
température maximale atteinte pendant l'essai à ce seuil.
""")

    c_a, c_b, c_c = st.columns(3)
    choix = [n for n, s_ in SCENARIOS.items() if s_ is not None]
    nom2 = c_a.selectbox("Faute étudiée", choix, key="faute_balayage")
    sc2 = SCENARIOS[nom2]
    unite = sc2["label"].split("(")[-1].rstrip(")")
    T_seuil2 = c_b.slider("Seuil de température admissible (°C)", 30, 90, 60, 1, key="seuil2")
    t_panne2 = None
    if nom2 == "Panne du refroidissement":
        t_panne2 = float(c_c.slider("Instant de la panne (min)", 0, 40, 10, 1, key="tp2"))
    with st.expander("Modifier le refroidissement du réacteur"):
        UA2 = st.slider("Efficacité de l'échange thermique UA (W/K)", 2.0, 30.0,
                        float(NOMINAL["UA"]), 0.5, key="ua2")
    st.caption(sc2["explication"])

    # --- simulations pour toute la plage du paramètre fautif
    valeurs = np.arange(sc2["mini"], sc2["maxi"] + sc2["pas"], sc2["pas"])
    courbes, Tmax = {}, []
    for x in valeurs:
        q = dict(NOMINAL)
        q["UA"] = UA2
        q.update(sc2["vers_param"](x))
        if t_panne2 is not None:
            q["t_panne"] = t_panne2
        tq, _, Tq, _, _ = simuler(q["n0"], q["n_eau"], q["T0"], q["Te"], q["UA"],
                                  q["t_panne"], q["perte"])
        courbes[x] = (tq, Tq)
        Tmax.append(Tq.max())
    Tmax = np.array(Tmax)
    depasse = Tmax > T_seuil2
    critique = valeurs[np.argmax(depasse)] if depasse.any() else None
    VERT, ROUGE = "#27ae60", "#c0392b"

    # --- 1. faisceau de courbes
    st.subheader("1. Plusieurs scénarios, deux comportements")
    st.markdown("""
Chaque courbe est un essai avec une valeur différente du paramètre fautif.
En vert, la température reste sous le seuil : la faute ne provoque pas d'emballement.
En rouge, elle le dépasse : la faute conduit à un emballement.
""")
    a_tracer = np.unique(np.round(np.linspace(sc2["mini"], sc2["maxi"], 7) / sc2["pas"]) * sc2["pas"])
    defaut_ex = VALEUR_NOMINALE[nom2]
    contexte = (nom2, T_seuil2, t_panne2, UA2)
    if st.session_state.get("contexte_prec") != contexte or st.session_state.get("examinee") not in list(valeurs):
        # nouvelle faute ou nouveaux réglages : on repart du fonctionnement normal
        st.session_state["examinee"] = defaut_ex
        st.session_state["contexte_prec"] = contexte
    examinee = st.select_slider("Valeur à examiner de plus près", options=list(valeurs),
                                format_func=lambda v: f"{v} {unite}", key="examinee")
    fig_f = go.Figure()
    for x in a_tracer:
        x = valeurs[np.argmin(abs(valeurs - x))]
        tq, Tq = courbes[x]
        danger = Tq.max() > T_seuil2
        fig_f.add_trace(go.Scatter(x=tq, y=Tq, mode="lines", name=f"{x} {unite}",
                                   line=dict(color=ROUGE if danger else VERT, width=1.8)))
    tq, Tq = courbes[examinee]
    fig_f.add_trace(go.Scatter(x=tq, y=Tq, mode="lines", name=f"examinée : {examinee} {unite}",
                               line=dict(color="black", width=4)))
    fig_f.add_hline(y=T_seuil2, line=dict(color="orange", dash="dash"),
                    annotation_text=f"seuil {T_seuil2} °C")
    fig_f.update_layout(height=450, xaxis_title="temps (min)", yaxis_title="Température (°C)",
                        legend_title_text=sc2["label"].split("(")[0].strip())
    st.plotly_chart(fig_f, width="stretch")

    i_ex = int(np.where(valeurs == examinee)[0][0])
    if depasse[i_ex]:
        t_dep = courbes[examinee][0][np.argmax(courbes[examinee][1] > T_seuil2)]
        st.error(f"Avec {examinee} {unite}, la température atteint {f'{Tmax[i_ex]:.1f}'.replace('.', ',')} °C : "
                 f"le seuil de {T_seuil2} °C est franchi à t = {f'{t_dep:.1f}'.replace('.', ',')} min. C'est un emballement.")
    else:
        st.success(f"Avec {examinee} {unite}, la température ne dépasse pas {f'{Tmax[i_ex]:.1f}'.replace('.', ',')} °C : "
                   f"elle reste sous le seuil de {T_seuil2} °C. Pas d'emballement.")

    # --- 2. température maximale en fonction de la faute
    st.subheader("2. La température maximale selon la gravité de la faute")
    st.markdown("""
On résume maintenant chaque essai par un seul point : sa température maximale.
On voit ainsi d'un coup d'œil quelles valeurs du paramètre sont sans danger et lesquelles sont dangereuses.
""")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=valeurs, y=Tmax, mode="lines+markers",
                              marker=dict(color=np.where(depasse, ROUGE, VERT), size=9),
                              line=dict(color="grey"), name="T max"))
    fig2.add_trace(go.Scatter(x=[examinee], y=[Tmax[i_ex]], mode="markers", name="valeur examinée",
                              marker=dict(color="rgba(0,0,0,0)", size=18,
                                          line=dict(color="black", width=3))))
    v_nom = VALEUR_NOMINALE[nom2]
    fig2.add_vline(x=v_nom, line=dict(color="blue", dash="dash"),
                   annotation_text="fonctionnement normal", annotation_position="bottom right")
    fig2.add_hline(y=T_seuil2, line=dict(color="orange", dash="dash"),
                   annotation_text=f"seuil {T_seuil2} °C")
    fig2.update_layout(xaxis_title=sc2["label"], yaxis_title="Température maximale (°C)",
                       height=480, showlegend=False)
    st.plotly_chart(fig2, width="stretch")
    st.caption("Point vert : la température reste sous le seuil. Point rouge : elle le dépasse. "
               "Cercle noir : la valeur examinée. Ligne bleue : la valeur en fonctionnement normal.")

    # --- 3. questions ouvertes sur la valeur critique
    st.subheader("3. À vous : quelle est la valeur critique ?")
    st.markdown(f"""
1. À partir de quelle valeur du paramètre fautif la température dépasse-t-elle le seuil de {T_seuil2} °C ?
   Cette valeur s'appelle la valeur critique.
2. Que se passe-t-il pour les valeurs plus petites que la valeur critique ? Et pour les valeurs plus grandes ?
3. Quelle est la marge de sécurité, c'est-à-dire l'écart entre la valeur critique et la valeur
   en fonctionnement normal ?
""")
    with st.expander("Un indice"):
        st.markdown("Faites varier la *valeur à examiner* en partant du fonctionnement nominal "
                    "(ligne bleue), puis augmentez-la petit à petit.")

    with st.expander("Questions pour aller plus loin"):
        st.markdown("""
1. Comparez les quatre fautes : laquelle a la plus petite marge de sécurité ? Laquelle est la plus dangereuse ?
2. Baissez le seuil de 60 °C à 50 °C. Que devient la valeur critique ? Pourquoi ?
3. Améliorez puis dégradez le refroidissement (UA). Comment la valeur critique évolue-t-elle ?
4. Pour la panne du refroidissement, changez l'instant de la panne. Une panne est-elle toujours dangereuse ?
5. La température maximale augmente-t-elle régulièrement avec la gravité de la faute, ou y a-t-il des
   valeurs où elle « décolle » plus vite ?
6. Juste au-dessus de la valeur critique, regardez la courbe examinée : à quel moment aurait-on pu
   deviner, avant le seuil, que l'essai allait mal finir ?
""")
