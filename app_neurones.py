"""
Découvrir un réseau de neurones
Application Streamlit — projet lycée (terminales spé maths / maths expertes)
Lancement local :  streamlit run app_neurones.py

Tous les calculs du réseau sont écrits ici avec numpy, sans bibliothèque
d'apprentissage : on peut lire chaque formule.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Découvrir un réseau de neurones", layout="wide")

VERT, ROUGE = "#27ae60", "#c0392b"
ECHELLE = [[0, "#f5b7b1"], [0.5, "#ffffff"], [1, "#abebc6"]]   # rouge -> blanc -> vert


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def fr(x, n=2):
    """Nombre écrit à la française."""
    return f"{x:.{n}f}".replace(".", ",")


def frl(x, n=2):
    """Nombre écrit à la française, dans une formule LaTeX (virgule sans espace)."""
    return f"{x:.{n}f}".replace(".", "{,}")


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
@st.cache_data
def donnees_examen():
    """Notes d'écrit et d'oral de 60 candidats. Admis si 0,6 × écrit + 0,4 × oral ≥ 10
    (avec un petit aléa : le jury n'est pas une machine)."""
    rng = np.random.default_rng(1)
    n = 60
    ecrit = np.round(rng.uniform(2, 19, n), 1)
    oral = np.round(rng.uniform(2, 19, n), 1)
    admis = ((0.6 * ecrit + 0.4 * oral + rng.normal(0, 0.6, n)) >= 10).astype(float)
    return ecrit, oral, admis


@st.cache_data
def donnees_plantes():
    """200 plantes. Une plante pousse bien si 15 °C ≤ température ≤ 30 °C et humidité ≥ 40 %."""
    rng = np.random.default_rng(2)
    n = 200
    temp = np.round(rng.uniform(0, 40, n), 1)
    hum = np.round(rng.uniform(0, 100, n), 0)
    pousse = ((temp >= 15) & (temp <= 30) & (hum >= 40)).astype(float)
    return temp, hum, pousse


ECRIT, ORAL, ADMIS = donnees_examen()
X_EXAMEN = np.c_[ECRIT / 20, ORAL / 20]          # notes ramenées entre 0 et 1
TEMP, HUM, POUSSE = donnees_plantes()
X_PLANTES = np.c_[TEMP / 40, HUM / 100]          # grandeurs ramenées entre 0 et 1


# ---------------------------------------------------------------------------
# Un neurone : calculs
# ---------------------------------------------------------------------------
def neurone(X, w1, w2, b):
    return sigmoide(w1 * X[:, 0] + w2 * X[:, 1] + b)


def cout(p, y):
    """Moyenne des carrés des écarts."""
    return float(np.mean((p - y) ** 2))


def score(p, y):
    return float(np.mean((p > 0.5) == (y > 0.5)))


def gradient_neurone(X, y, w1, w2, b):
    """Dérivées du coût par rapport à w1, w2 et b."""
    p = neurone(X, w1, w2, b)
    d = 2 * (p - y) * p * (1 - p) / len(y)
    return float(d @ X[:, 0]), float(d @ X[:, 1]), float(d.sum())


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------
def carte_examen(w1, w2, b, titre=""):
    """Fond coloré selon la réponse du neurone, points des candidats, frontière."""
    g = np.linspace(0, 20, 121)
    E, O = np.meshgrid(g, g)
    P = sigmoide(w1 * E / 20 + w2 * O / 20 + b)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=g, y=g, z=P, colorscale=ECHELLE, zmin=0, zmax=1, opacity=0.8,
                             colorbar=dict(title="réponse<br>du neurone"), hoverinfo="skip"))
    if abs(w2) > 1e-9:
        xs = np.array([0, 20])
        ys = -(w1 * xs / 20 + b) / w2 * 20
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="black", dash="dash"),
                                 name="frontière (réponse = 0,5)"))
    elif abs(w1) > 1e-9:
        xv = -b / w1 * 20
        fig.add_trace(go.Scatter(x=[xv, xv], y=[0, 20], mode="lines",
                                 line=dict(color="black", dash="dash"), name="frontière (réponse = 0,5)"))
    for val, nom, coul, symb in [(1, "admis", VERT, "circle"), (0, "refusé", ROUGE, "x")]:
        m = ADMIS == val
        fig.add_trace(go.Scatter(x=ECRIT[m], y=ORAL[m], mode="markers", name=nom,
                                 marker=dict(color=coul, size=9, symbol=symb,
                                             line=dict(width=1, color="black"))))
    fig.update_layout(height=520, title=titre, xaxis=dict(title="note d'écrit", range=[0, 20]),
                      yaxis=dict(title="note d'oral", range=[0, 20], scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.15))
    return fig


# ---------------------------------------------------------------------------
# Réseau à une couche cachée : calculs
# ---------------------------------------------------------------------------
def init_reseau(h, graine):
    r = np.random.default_rng(graine)
    return dict(W1=r.normal(0, 1, (h, 2)), b1=r.normal(0, 1, h), W2=r.normal(0, 1, h), b2=0.0)


def propager(R, X):
    A = sigmoide(X @ R["W1"].T + R["b1"])        # sorties des neurones cachés
    p = sigmoide(A @ R["W2"] + R["b2"])          # sortie du réseau
    return A, p


def entrainer(R, X, y, pas, iterations):
    n = len(y)
    for _ in range(iterations):
        A, p = propager(R, X)
        d2 = 2 * (p - y) * p * (1 - p) / n
        gW2, gb2 = A.T @ d2, d2.sum()
        dA = np.outer(d2, R["W2"]) * A * (1 - A)
        gW1, gb1 = dA.T @ X, dA.sum(0)
        R["W1"] -= pas * gW1
        R["b1"] -= pas * gb1
        R["W2"] -= pas * gW2
        R["b2"] -= pas * gb2
    return R


def matrice_latex(M):
    M = np.atleast_2d(M)
    lignes = [" & ".join(frl(v) for v in ligne) for ligne in M]
    return r"\begin{pmatrix}" + r" \\ ".join(lignes) + r"\end{pmatrix}"


# ===========================================================================
# Interface
# ===========================================================================
st.title("Découvrir un réseau de neurones")

o1, o2, o3, o4, o5 = st.tabs(["1. Un neurone", "2. La sigmoïde", "3. Mesurer l'erreur",
                              "4. Apprendre tout seul", "5. La couche cachée"])

# ------------------------------------------------------------------ Onglet 1
with o1:
    st.markdown("""
Voici 60 candidats fictifs à un examen. Chacun a une note d'écrit et une note d'oral. Un jury,
fictif lui aussi, a décidé qui est admis : sur le graphique, les ronds verts sont admis, les croix
rouges sont refusées.

Le jury applique une règle secrète, qui dépend des deux notes. Comme tout jury humain, il ne
l'applique pas toujours de façon parfaitement mécanique.

Votre mission : construire une machine, un neurone, qui prenne les deux notes et réponde « admis »
ou « refusé » en se trompant le moins possible... et retrouver ainsi la règle secrète du jury.
""")
    c_g, c_d = st.columns([1, 2])
    with c_g:
        st.markdown("Le neurone ramène d'abord les notes entre 0 et 1 :")
        st.latex(r"x_1 = \frac{\text{écrit}}{20} \qquad x_2 = \frac{\text{oral}}{20}")
        st.markdown("Il calcule ensuite un nombre $z$ à l'aide de deux poids $w_1$, $w_2$ et d'un biais $b$ :")
        st.latex(r"z = w_1\,x_1 + w_2\,x_2 + b")
        st.markdown("Enfin, il transforme $z$ en une réponse entre 0 et 1 avec la fonction sigmoïde :")
        st.latex(r"\text{réponse} = \sigma(z) = \frac{1}{1 + e^{-z}}")
        st.markdown("Si la réponse dépasse 0,5, le neurone dit « admis ».")
        st.divider()
        w1 = st.slider("Poids w₁ (écrit)", -20.0, 20.0, 1.0, 0.5, key="w1")
        w2 = st.slider("Poids w₂ (oral)", -20.0, 20.0, 1.0, 0.5, key="w2")
        b = st.slider("Biais b", -20.0, 20.0, 0.0, 0.5, key="b")
        p = neurone(X_EXAMEN, w1, w2, b)
        st.metric("Bonnes réponses", f"{score(p, ADMIS) * 100:.0f} %")
    with c_d:
        st.plotly_chart(carte_examen(w1, w2, b), width="stretch", key="graphique_1")
        st.caption("Fond vert : le neurone répond « admis ». Fond rouge : « refusé ». "
                   "Ligne pointillée : là où la réponse vaut exactement 0,5.")

    with st.expander("Le calcul pour un candidat"):
        i = st.number_input("Numéro du candidat", 1, len(ECRIT), 1) - 1
        z = w1 * X_EXAMEN[i, 0] + w2 * X_EXAMEN[i, 1] + b
        st.latex(rf"x_1 = \frac{{{frl(ECRIT[i], 1)}}}{{20}} = {frl(X_EXAMEN[i, 0], 3)} \qquad "
                 rf"x_2 = \frac{{{frl(ORAL[i], 1)}}}{{20}} = {frl(X_EXAMEN[i, 1], 3)}")
        st.latex(rf"z = {frl(w1, 1)} \times {frl(X_EXAMEN[i, 0], 3)} + {frl(w2, 1)} \times "
                 rf"{frl(X_EXAMEN[i, 1], 3)} + ({frl(b, 1)}) = {frl(z, 3)}")
        st.latex(rf"\sigma(z) = {frl(sigmoide(z), 3)}")
        rep = sigmoide(z)
        verdict = "admis" if rep > 0.5 else "refusé"
        comparaison = ">" if rep > 0.5 else "≤"
        vrai = "admis" if ADMIS[i] == 1 else "refusé"
        st.markdown(f"Réponse du neurone : {fr(rep, 3)} {comparaison} 0,5, donc le neurone dit « {verdict} ».  \n"
                    f"Décision du jury : « {vrai} ».  \n"
                    + ("C'est une bonne réponse du neurone." if verdict == vrai
                       else "Le neurone se trompe pour ce candidat."))

    with st.expander("Défis"):
        st.markdown("""
1. Trouvez des réglages qui donnent au moins 90 % de bonnes réponses.
2. Que se passe-t-il quand on augmente $w_1$ ? Quand on augmente $b$ ?
3. La frontière est une droite. Écrivez son équation en fonction de $w_1$, $w_2$ et $b$.
4. Le jury donne-t-il plus d'importance à l'écrit ou à l'oral ? Comment le voir sur vos réglages ?
""")

# ------------------------------------------------------------------ Onglet 2
with o2:
    c_g, c_d = st.columns(2)
    with c_g:
        st.markdown("La fonction sigmoïde transforme n'importe quel nombre $z$ en un nombre entre 0 et 1.")
        st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
        z0 = st.slider("Valeur de z", -8.0, 8.0, 0.0, 0.1)
        zz = np.linspace(-8, 8, 400)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=zz, y=sigmoide(zz), mode="lines", line=dict(width=3), name="σ(z)"))
        fig.add_trace(go.Scatter(x=[z0], y=[sigmoide(z0)], mode="markers",
                                 marker=dict(size=13, color="black"), name="point choisi"))
        fig.add_hline(y=0.5, line=dict(color="grey", dash="dot"))
        fig.update_layout(height=400, xaxis_title="z", yaxis_title="σ(z)", showlegend=False)
        st.plotly_chart(fig, width="stretch", key="graphique_2")
        st.markdown(f"σ({fr(z0, 1)}) = {fr(sigmoide(z0), 3)}")
    with c_d:
        st.markdown("""
On reprend les réglages de l'onglet 1 et on les multiplie tous par un même nombre $k$.
""")
        k = st.select_slider("Multiplier w₁, w₂ et b par", [0.25, 0.5, 1, 2, 5, 10], value=1)
        w1, w2, b = st.session_state["w1"], st.session_state["w2"], st.session_state["b"]
        st.plotly_chart(carte_examen(k * w1, k * w2, k * b), width="stretch", key="graphique_3")
    with st.expander("Questions"):
        st.markdown("""
1. Quelles sont les limites de $\\sigma(z)$ quand $z$ tend vers $+\\infty$ et vers $-\\infty$ ?
2. Pour quelle valeur de $z$ a-t-on $\\sigma(z) = 0{,}5$ ? Pourquoi la frontière correspond-elle à $z = 0$ ?
3. Étudiez le sens de variation de $\\sigma$. Calculez $\\sigma'(z)$ et montrez que $\\sigma'(z) = \\sigma(z)\\,(1 - \\sigma(z))$.
4. Quand on multiplie les réglages par $k$, la frontière bouge-t-elle ? Qu'est-ce qui change ?
""")

# ------------------------------------------------------------------ Onglet 3
with o3:
    w1, w2, b = st.session_state["w1"], st.session_state["w2"], st.session_state["b"]
    p = neurone(X_EXAMEN, w1, w2, b)
    st.markdown("""
Pour qu'une machine puisse s'améliorer, il faut d'abord mesurer à quel point elle se trompe.
Pour chaque candidat, on compare la réponse du neurone (entre 0 et 1) à la bonne réponse
(1 pour admis, 0 pour refusé), et on élève l'écart au carré. Le coût est la moyenne de ces carrés :
""")
    st.latex(r"C = \frac{1}{n} \sum_{i=1}^{n} \left(\text{réponse}_i - \text{bonne réponse}_i\right)^2")
    c_g, c_d = st.columns([1, 1])
    with c_g:
        st.markdown("Les cinq premiers candidats, avec les réglages de l'onglet 1 :")
        lignes = ("| Candidat | Écrit | Oral | Réponse du neurone | Bonne réponse (1 = admis, 0 = refusé) "
                  "| Écart au carré |\n|---|---|---|---|---|---|\n")
        for i in range(5):
            lignes += (f"| {i + 1} | {fr(ECRIT[i], 1)} | {fr(ORAL[i], 1)} | {fr(p[i], 3)} | "
                       f"{int(ADMIS[i])} | {fr((p[i] - ADMIS[i]) ** 2, 3)} |\n")
        st.markdown(lignes)
        st.metric("Coût sur les 60 candidats", fr(cout(p, ADMIS), 3))
    with c_d:
        st.markdown("On garde $w_2$ et $b$ fixés, et on fait varier $w_1$ seul :")
        ww = np.linspace(-20, 20, 401)
        cc = [cout(neurone(X_EXAMEN, v, w2, b), ADMIS) for v in ww]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ww, y=cc, mode="lines", line=dict(width=3), name="coût"))
        fig.add_trace(go.Scatter(x=[w1], y=[cout(p, ADMIS)], mode="markers",
                                 marker=dict(size=13, color="black"), name="réglage actuel"))
        fig.update_layout(height=420, xaxis_title="w₁", yaxis_title="coût C", showlegend=False)
        st.plotly_chart(fig, width="stretch", key="graphique_4")
    with st.expander("Questions"):
        st.markdown("""
1. Pourquoi élève-t-on les écarts au carré au lieu de simplement les additionner ?
2. Que vaudrait le coût d'un neurone parfait ?
3. Sur la courbe, pour quelle valeur de $w_1$ le coût est-il le plus petit ? Que vaut la dérivée en ce point ?
4. Changez $w_2$ ou $b$ dans l'onglet 1 : la courbe change-t-elle ? Le meilleur $w_1$ aussi ?
""")

# ------------------------------------------------------------------ Onglet 4
with o4:
    if "gd" not in st.session_state:
        st.session_state["gd"] = dict(w1=0.0, w2=0.0, b=0.0, historique=[])

    st.markdown("""
Au lieu de chercher les réglages à la main, on laisse le neurone les trouver. À chaque pas, il
regarde dans quel sens le coût diminue (c'est le rôle des dérivées) et il modifie un peu chaque réglage
dans ce sens. C'est la descente de gradient :
""")
    st.latex(r"w_1 \leftarrow w_1 - \eta\,\frac{\partial C}{\partial w_1} \qquad "
             r"w_2 \leftarrow w_2 - \eta\,\frac{\partial C}{\partial w_2} \qquad "
             r"b \leftarrow b - \eta\,\frac{\partial C}{\partial b}")
    st.markdown("Le nombre $\\eta$ s'appelle le pas d'apprentissage.")

    gd = st.session_state["gd"]
    c_g, c_d = st.columns([1, 2])
    with c_g:
        pas = st.select_slider("Pas d'apprentissage η", [0.5, 1, 2, 5, 10, 20, 50, 100, 300], value=10)
        b1c, b2c, b3c = st.columns(3)
        nb = 0
        if b1c.button("1 pas"):
            nb = 1
        if b2c.button("10 pas"):
            nb = 10
        if b3c.button("100 pas"):
            nb = 100
        r1, r2 = st.columns(2)
        if r1.button("Repartir de zéro"):
            gd.update(w1=0.0, w2=0.0, b=0.0, historique=[])
        if r2.button("Repartir de mes réglages"):
            gd.update(w1=st.session_state["w1"], w2=st.session_state["w2"],
                      b=st.session_state["b"], historique=[])
        if not gd["historique"]:
            gd["historique"].append(cout(neurone(X_EXAMEN, gd["w1"], gd["w2"], gd["b"]), ADMIS))
        for _ in range(nb):
            g1, g2, gb = gradient_neurone(X_EXAMEN, ADMIS, gd["w1"], gd["w2"], gd["b"])
            gd["w1"] -= pas * g1
            gd["w2"] -= pas * g2
            gd["b"] -= pas * gb
            gd["historique"].append(cout(neurone(X_EXAMEN, gd["w1"], gd["w2"], gd["b"]), ADMIS))

        p = neurone(X_EXAMEN, gd["w1"], gd["w2"], gd["b"])
        st.markdown(f"Nombre de pas effectués : {len(gd['historique']) - 1}")
        st.markdown(f"w₁ = {fr(gd['w1'])}  \nw₂ = {fr(gd['w2'])}  \nb = {fr(gd['b'])}")
        m1, m2 = st.columns(2)
        m1.metric("Coût", fr(gd["historique"][-1], 3))
        m2.metric("Bonnes réponses", f"{score(p, ADMIS) * 100:.0f} %")
        g1, g2, gb = gradient_neurone(X_EXAMEN, ADMIS, gd["w1"], gd["w2"], gd["b"])
        with st.expander("Les dérivées au point actuel"):
            st.latex(rf"\frac{{\partial C}}{{\partial w_1}} = {frl(g1, 4)} \qquad "
                     rf"\frac{{\partial C}}{{\partial w_2}} = {frl(g2, 4)} \qquad "
                     rf"\frac{{\partial C}}{{\partial b}} = {frl(gb, 4)}")
    with c_d:
        st.plotly_chart(carte_examen(gd["w1"], gd["w2"], gd["b"]), width="stretch", key="graphique_5")
        fig = go.Figure(go.Scatter(y=gd["historique"], mode="lines+markers", marker=dict(size=4)))
        fig.update_layout(height=280, xaxis_title="nombre de pas", yaxis_title="coût",
                          margin=dict(t=20))
        st.plotly_chart(fig, width="stretch", key="graphique_6")
    with st.expander("Questions"):
        st.markdown("""
1. Repartez de zéro avec η = 10 et faites 100 pas plusieurs fois. Comment évoluent le coût et la frontière ?
2. Recommencez avec η = 0,5 puis avec η = 300. Que remarquez-vous ? Quel pas choisiriez-vous ?
3. Après un long apprentissage, calculez le rapport $w_1 / w_2$. D'après vous, le jury donne-t-il
   plus d'importance à l'écrit ou à l'oral ? Proposez une règle pour le jury.
4. Pourquoi le neurone n'atteint-il jamais 100 % de bonnes réponses ?
""")

    with st.expander("Révéler la règle secrète du jury"):
        st.markdown("""
Pour chaque candidat, le jury calcule la moyenne pondérée
""")
        st.latex(r"0{,}6 \times \text{écrit} + 0{,}4 \times \text{oral}")
        st.markdown("""
puis y ajoute un petit nombre tiré au hasard, en général compris entre −1 et 1 : c'est sa part
d'appréciation humaine. Le candidat est admis si le résultat atteint 10.

Loin de 10, le hasard ne change rien. Près de 10, il peut faire basculer la décision : c'est pourquoi
aucun réglage du neurone ne peut donner 100 % de bonnes réponses.

Remarque : la règle « admis si $0{,}6 \\times \\text{écrit} + 0{,}4 \\times \\text{oral} \\geq 10$ »
correspond, pour le neurone, aux réglages $w_1 = 12$, $w_2 = 8$, $b = -10$, et à tous leurs multiples par un même nombre positif.
Vérifiez-le !
""")

# ------------------------------------------------------------------ Onglet 5
with o5:
    st.markdown("""
Voici 200 plantes fictives. Pour chacune, on connaît la température et l'humidité de l'air, et si
elle pousse bien (rond vert) ou non (croix rouge). Peut-on séparer les ronds verts des croix rouges
avec une seule droite ?

On place maintenant plusieurs neurones côte à côte : c'est une couche cachée. Chaque neurone caché
reçoit la température et l'humidité (ramenées entre 0 et 1) et trace sa propre droite. Un dernier
neurone, le neurone de sortie, combine leurs réponses.
""")
    c_g, c_d = st.columns([1, 2])
    with c_g:
        h = st.slider("Nombre de neurones cachés", 1, 6, 1)
        pas5 = st.select_slider("Pas d'apprentissage η", [1, 2, 5, 10, 20, 50], value=10, key="pas5")
        graine = st.number_input("Tirage des poids de départ", 0, 99, 0,
                                 help="Changer ce nombre change les poids tirés au hasard au départ.")
        cle = (h, graine)
        if st.session_state.get("cle_reseau") != cle:
            st.session_state["reseau"] = init_reseau(h, graine)
            st.session_state["histo5"] = []
            st.session_state["cle_reseau"] = cle
        R = st.session_state["reseau"]
        e1, e2 = st.columns(2)
        nb = 0
        if e1.button("Entraîner (1 000 itérations)"):
            nb = 1000
        if e2.button("Recommencer"):
            st.session_state["reseau"] = R = init_reseau(h, graine)
            st.session_state["histo5"] = []
        histo = st.session_state["histo5"]
        if not histo:
            histo.append(cout(propager(R, X_PLANTES)[1], POUSSE))
        for _ in range(nb // 100):
            entrainer(R, X_PLANTES, POUSSE, pas5, 100)
            histo.append(cout(propager(R, X_PLANTES)[1], POUSSE))
        A, p = propager(R, X_PLANTES)
        st.markdown(f"Itérations effectuées : {(len(histo) - 1) * 100}")
        m1, m2 = st.columns(2)
        m1.metric("Coût", fr(histo[-1], 3))
        m2.metric("Bonnes réponses", f"{score(p, POUSSE) * 100:.0f} %")
        voir_droites = st.checkbox("Afficher les droites des neurones cachés", value=True)

    with c_d:
        gt = np.linspace(0, 40, 121)
        gh = np.linspace(0, 100, 121)
        TT, HH = np.meshgrid(gt, gh)
        grille = np.c_[TT.ravel() / 40, HH.ravel() / 100]
        P = propager(R, grille)[1].reshape(TT.shape)
        fig = go.Figure()
        fig.add_trace(go.Heatmap(x=gt, y=gh, z=P, colorscale=ECHELLE, zmin=0, zmax=1, opacity=0.8,
                                 colorbar=dict(title="réponse<br>du réseau"), hoverinfo="skip"))
        if voir_droites:
            for j in range(h):
                a1, a2 = R["W1"][j]
                c0 = R["b1"][j]
                if abs(a2) > 1e-9:
                    xs = np.array([0, 40])
                    ys = -(a1 * xs / 40 + c0) / a2 * 100
                else:
                    xs = np.array([-c0 / a1 * 40] * 2)
                    ys = np.array([0, 100])
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"neurone caché {j + 1}",
                                         line=dict(dash="dot", width=2)))
        for val, nom, coul, symb in [(1, "pousse bien", VERT, "circle"), (0, "pousse mal", ROUGE, "x")]:
            m = POUSSE == val
            fig.add_trace(go.Scatter(x=TEMP[m], y=HUM[m], mode="markers", name=nom,
                                     marker=dict(color=coul, size=7, symbol=symb,
                                                 line=dict(width=1, color="black"))))
        fig.update_layout(height=520, xaxis=dict(title="température (°C)", range=[0, 40]),
                          yaxis=dict(title="humidité (%)", range=[0, 100]),
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch", key="graphique_7")
        fig2 = go.Figure(go.Scatter(x=np.arange(len(histo)) * 100, y=histo, mode="lines"))
        fig2.update_layout(height=250, xaxis_title="itérations", yaxis_title="coût", margin=dict(t=20))
        st.plotly_chart(fig2, width="stretch", key="graphique_8")

    with st.expander("Le schéma du réseau"):
        dot = ["digraph G {", "rankdir=LR;", 'node [shape=circle, style=filled, fillcolor="#eef3fb"];',
               'x1 [label="température"]; x2 [label="humidité"]; s [label="sortie", fillcolor="#d5f5e3"];']
        for j in range(h):
            dot.append(f'c{j} [label="caché {j + 1}"];')
            for i_, x in enumerate(["x1", "x2"]):
                dot.append(f'{x} -> c{j} [label="{fr(R["W1"][j, i_], 1)}"];')
            dot.append(f'c{j} -> s [label="{fr(R["W2"][j], 1)}"];')
        dot.append("}")
        st.graphviz_chart("\n".join(dot))
        st.caption("Les nombres sur les flèches sont les poids.")

    with st.expander("Le calcul avec des matrices (maths expertes)"):
        st.markdown("Si $x$ est le vecteur des deux entrées, la couche cachée calcule :")
        st.latex(r"a = \sigma\left(W_1\, x + b_1\right) \qquad \text{puis} \qquad "
                 r"\text{réponse} = \sigma\left(W_2\, a + b_2\right)")
        st.latex(r"W_1 = " + matrice_latex(R["W1"]) + r"\qquad b_1 = " + matrice_latex(R["b1"].reshape(-1, 1)))
        st.latex(r"W_2 = " + matrice_latex(R["W2"].reshape(1, -1)) + rf"\qquad b_2 = {frl(R['b2'])}")
        st.markdown("Ici, $\\sigma$ s'applique à chaque coordonnée du vecteur.")

    with st.expander("Questions"):
        st.markdown("""
1. Avec 1 neurone caché, entraînez le réseau longtemps. Quel pourcentage de bonnes réponses obtenez-vous ?
   Quelle proportion des plantes poussent mal ? Que remarquez-vous ?
2. Recommencez avec 2, 3, puis 4 neurones cachés. Combien en faut-il pour bien séparer les plantes ? Pourquoi ce nombre ?
3. Affichez les droites des neurones cachés : que représente chacune d'elles ?
4. Changez le tirage des poids de départ. Le résultat est-il toujours le même ?
5. Combien le réseau a-t-il de nombres à régler (poids et biais) avec 3 neurones cachés ?
""")

# ------------------------------------------------------------------ Pour aller plus loin
st.divider()
st.markdown("""
Pour aller plus loin, vous pouvez lire le livre
[Deepmath : mathématiques des réseaux de neurones](https://exo7.emath.fr/cours/livre-deepmath.pdf)
d'Arnaud Bodin et François Recher (collection Exo7, en accès libre), en particulier :

- le chapitre 5, « Réseau de neurones » (onglets 1, 2 et 5) ;
- le chapitre 7, « Gradient » (onglet 3) ;
- le chapitre 8, « Descente de gradient » (onglet 4) ;
- le chapitre 9, « Rétropropagation » (comment le réseau de l'onglet 5 calcule ses dérivées).
""")
