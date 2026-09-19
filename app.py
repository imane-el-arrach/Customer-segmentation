"""Dashboard decisionnel base sur l'artefact RFM/K-Means du notebook."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config("SegmentIQ | Customer Intelligence", "◈", layout="wide")

DATA_PATH = Path(__file__).with_name("full_clustering_preprocessed.csv")
COLORS = {"DELIGHT":"#7C3AED", "PAMPER":"#DB2777", "REWARD":"#F59E0B",
          "RETAIN":"#0EA5A4", "UPSELL":"#2563EB", "NURTURE":"#64748B", "RE-ENGAGE":"#EF4444"}
PLAYBOOKS = {
    "DELIGHT": ("Préserver une relation à très forte valeur", "Accès anticipé, service prioritaire et offre VIP personnalisée.", "Gestionnaire de compte ou e-mail personnalisé"),
    "PAMPER": ("Fidéliser les clients à forte contribution", "Programme de reconnaissance, avantages premium et surprises ciblées.", "E-mail personnalisé"),
    "REWARD": ("Récompenser la fréquence d'achat", "Points fidélité, bundles et accès en avant-première.", "E-mail ou programme de fidélité"),
    "RETAIN": ("Maintenir la régularité d'achat", "Recommandations complémentaires et rappels légers.", "E-mail automatisé"),
    "UPSELL": ("Développer le panier moyen de clients actifs", "Bundles premium et produits complémentaires adaptés à l'historique.", "E-mail ou recommandation sur site"),
    "NURTURE": ("Faire progresser les clients à potentiel", "Séquence de découverte, best-sellers et incentive de second achat.", "E-mail de nurturing"),
    "RE-ENGAGE": ("Réactiver les clients devenus inactifs", "Campagne win-back avec une offre pertinente et limitée.", "E-mail de réactivation"),
}

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Charge le resultat réel du clustering; l'app ne fabrique aucune donnée."""
    df = pd.read_csv(path).rename(columns={"Customer ID":"CustomerID", "ClusterLabel":"Segment"})
    required = {"CustomerID", "MonetaryValue", "Frequency", "LastInvoiceDate", "Recency", "Segment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {', '.join(sorted(missing))}")
    for col in ["MonetaryValue", "Frequency", "Recency"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["LastInvoiceDate"] = pd.to_datetime(df["LastInvoiceDate"], errors="coerce")
    df = df.dropna(subset=["CustomerID", "MonetaryValue", "Frequency", "Recency", "Segment"])
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    return df

def euros(value): return f"€{value:,.0f}".replace(",", " ")
def pct(value): return f"{value:.1%}".replace(".", ",")
def summary(df):
    result = df.groupby("Segment", as_index=False).agg(
        Clients=("CustomerID", "nunique"), CA=("MonetaryValue", "sum"),
        Panier_moyen=("MonetaryValue", "mean"), Frequence=("Frequency", "mean"), Recence=("Recency", "mean"))
    result["Part_CA"] = result["CA"] / result["CA"].sum()
    return result.sort_values("CA", ascending=False)

st.markdown("""<style>
.stApp{background:#F7F8FC;color:#172033}[data-testid="stSidebar"]{background:#111827}
[data-testid="stSidebar"] *{color:#F9FAFB}.hero{padding:.2rem 0 1rem}.eyebrow{color:#7C3AED;font-size:.78rem;font-weight:700;letter-spacing:.12em}
.hero h1{color:#111827;font-size:2.5rem;letter-spacing:-.05em;margin:.1rem 0}.hero p{color:#64748B;margin:0;font-size:1.05rem}
.insight{background:#181B31;color:#F8FAFC;border-radius:14px;padding:1rem 1.25rem}.insight strong{color:#C4B5FD}
div[data-testid="stMetric"]{background:#fff;border:1px solid #E7EAF1;padding:.9rem;border-radius:12px}
.stTabs [data-baseweb="tab-list"]{gap:1.25rem}.stTabs [data-baseweb="tab"]{padding-left:0;padding-right:0;font-weight:600}
</style>""", unsafe_allow_html=True)

try:
    df = load_data(DATA_PATH)
except (FileNotFoundError, ValueError) as error:
    st.error(f"Impossible de charger l'artefact de clustering : {error}")
    st.stop()

segments = [segment for segment in PLAYBOOKS if segment in df.Segment.unique()]
with st.sidebar:
    st.markdown("# SegmentIQ")
    st.caption("Customer intelligence • RFM + K-Means")
    st.divider(); st.markdown("#### Périmètre d'analyse")
    selected = st.multiselect("Segments", segments, default=segments, label_visibility="collapsed")
    limit = st.slider("Récence maximale (jours)", 0, int(df.Recency.max()), int(df.Recency.max()))
    st.divider();

filtered = df[df.Segment.isin(selected) & df.Recency.le(limit)].copy()
st.markdown("""<div class="hero"><div class="eyebrow">CUSTOMER INTELLIGENCE</div><h1>De la segmentation aux décisions.</h1><p>Une lecture actionnable des comportements d'achat issus de l'analyse RFM.</p></div>""", unsafe_allow_html=True)
if filtered.empty:
    st.warning("Aucun client ne correspond aux filtres actuels."); st.stop()

stats = summary(filtered)
revenue = filtered.MonetaryValue.sum()
top20 = filtered.nlargest(max(1, int(len(filtered)*.2)), "MonetaryValue").MonetaryValue.sum()
active = filtered.loc[filtered.Recency.le(30), "CustomerID"].nunique()
reengage = filtered.loc[filtered.Segment.eq("RE-ENGAGE"), "MonetaryValue"].sum()
kpis = st.columns(4)
kpis[0].metric("Clients analysés", f"{filtered.CustomerID.nunique():,}".replace(",", " "))
kpis[1].metric("CA historique observé", euros(revenue))
kpis[2].metric("Clients actifs (≤ 30 j)", f"{active:,}".replace(",", " "))
kpis[3].metric("CA des 20 % premiers clients", pct(top20/revenue))
lead = stats.iloc[0]
reengage_text = f"Le vivier <strong>RE-ENGAGE</strong> représente {euros(reengage)} à réactiver." if reengage else "Le segment RE-ENGAGE n'est pas inclus dans les filtres."
st.markdown(f"<div class='insight'>Priorité business : <strong>{lead.Segment}</strong> concentre {pct(lead.Part_CA)} du CA observé avec {lead.Clients:,} clients. {reengage_text}</div>".replace(",", " "), unsafe_allow_html=True)

t1, t2, t3, t4 = st.tabs(["Vue d'ensemble", "Profils segments", "Plan de campagne", "Explorateur clients"])
with t1:
    left, right = st.columns((1.05,.95), gap="large")
    with left:
        fig = px.bar(stats.sort_values("CA"), x="CA", y="Segment", orientation="h", color="Segment", color_discrete_map=COLORS, text="Part_CA", title="Contribution au chiffre d'affaires par segment")
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, height=390, margin=dict(l=0,r=35,t=55,b=10), xaxis_title="CA historique")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.scatter(filtered, x="Recency", y="Frequency", size="MonetaryValue", color="Segment", color_discrete_map=COLORS, hover_data={"CustomerID":True,"MonetaryValue":":,.2f","Recency":True,"Frequency":True}, title="Engagement : récence, fréquence et valeur", labels={"Recency":"Jours depuis le dernier achat", "Frequency":"Nombre d'achats"})
        fig.update_layout(height=390, margin=dict(l=0,r=0,t=55,b=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    st.markdown("#### Lecture rapide")
    c1,c2,c3 = st.columns(3); active_segment=stats.loc[stats.Recence.idxmin()]; frequent=stats.loc[stats.Frequence.idxmax()]
    c1.info(f"**À protéger**\n\n{lead.Segment} est le premier contributeur de valeur.")
    c2.info(f"**Le plus actif**\n\n{active_segment.Segment} achète en moyenne tous les {active_segment.Recence:.0f} jours.")
    c3.info(f"**Le plus fréquent**\n\n{frequent.Segment} affiche {frequent.Frequence:.1f} achats en moyenne.")

with t2:
    st.markdown("#### Comparer les profils RFM")
    display = stats.rename(columns={"CA":"CA historique", "Panier_moyen":"Panier moyen", "Frequence":"Fréquence moyenne", "Recence":"Récence moyenne", "Part_CA":"Part du CA"}).copy()
    display["CA historique"] = display["CA historique"].map(euros); display["Panier moyen"] = display["Panier moyen"].map(euros); display["Part du CA"] = display["Part du CA"].map(pct)
    display["Fréquence moyenne"] = display["Fréquence moyenne"].map(lambda x:f"{x:.1f}"); display["Récence moyenne"] = display["Récence moyenne"].map(lambda x:f"{x:.0f} j")
    st.dataframe(display, width="stretch", hide_index=True)
    chosen = st.selectbox("Choisir un segment à analyser", stats.Segment.tolist())
    data = filtered[filtered.Segment.eq(chosen)]; profile = summary(data).iloc[0]; goal, action, channel = PLAYBOOKS[chosen]
    left,right=st.columns((1.05,.95), gap="large")
    with left:
        a,b,c=st.columns(3); a.metric("Clients", f"{profile.Clients:,}".replace(","," ")); b.metric("CA historique", euros(profile.CA)); c.metric("Récence moyenne", f"{profile.Recence:.0f} jours")
        fig=px.histogram(data,x="MonetaryValue",nbins=35,color_discrete_sequence=[COLORS[chosen]],title=f"Distribution de valeur — {chosen}",labels={"MonetaryValue":"Valeur monétaire"})
        fig.update_layout(height=300,margin=dict(l=0,r=0,t=55,b=10)); st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Action recommandée"); st.success(f"**Objectif :** {goal}"); st.write(action); st.caption(f"Canal conseillé : {channel}")

with t3:
    st.markdown("#### Préparer une campagne ciblée")
    target=st.selectbox("Segment cible", segments, key="campaign"); audience=filtered[filtered.Segment.eq(target)].copy(); goal,action,channel=PLAYBOOKS[target]
    left,right=st.columns((.8,1.2),gap="large")
    with left:
        st.metric("Audience sélectionnée", f"{audience.CustomerID.nunique():,}".replace(","," ")); st.metric("Valeur historique", euros(audience.MonetaryValue.sum())); st.metric("Inactivité moyenne", f"{audience.Recency.mean():.0f} jours")
    with right:
        st.markdown(f"**Objectif :** {goal}\n\n**Action :** {action}\n\n**Canal :** {channel}")
    export=audience[["CustomerID","Segment","Recency","Frequency","MonetaryValue","LastInvoiceDate"]].sort_values("MonetaryValue",ascending=False)
    st.download_button("Télécharger l'audience de campagne (CSV)", export.to_csv(index=False).encode("utf-8-sig"), f"audience_{target.lower().replace('-','_')}.csv", "text/csv")

with t4:
    st.markdown("#### Identifier les clients prioritaires")
    st.caption("Triée par valeur monétaire observée. Les indicateurs décrivent l'historique, pas une prédiction future.")
    st.dataframe(filtered[["CustomerID","Segment","Recency","Frequency","MonetaryValue","LastInvoiceDate"]].sort_values("MonetaryValue",ascending=False), width="stretch", hide_index=True, column_config={"MonetaryValue":st.column_config.NumberColumn("Valeur monétaire",format="€%.2f"),"Recency":st.column_config.NumberColumn("Récence (jours)"),"Frequency":st.column_config.NumberColumn("Fréquence"),"LastInvoiceDate":st.column_config.DatetimeColumn("Dernier achat",format="DD/MM/YYYY")})

st.divider(); st.caption("Méthode : RFM (récence, fréquence, valeur monétaire) + K-Means. Source : résultat prétraité du notebook de clustering.")
