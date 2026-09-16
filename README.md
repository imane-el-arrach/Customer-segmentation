# 📊 Customer Segmentation — RFM Analysis & K-Means Clustering

![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Viz-Plotly%203D-3F4F75?logo=plotly&logoColor=white)
![Status](https://img.shields.io/badge/status-stable-success)

Une application interactive de **segmentation client** basée sur les données de transactions du dataset **Online Retail II** (UCI Machine Learning Repository), combinant **analyse RFM** (Recency, Frequency, Monetary) et **clustering K-Means**, avec gestion explicite des outliers pour ne pas fausser les résultats.

🔗 **Démo live** : [Tester l'application](https://customer-segmentation-f8fkkbcmwcput4fsbpbahq.streamlit.app/)
📁 **Dataset** : [Online Retail II — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) *(fichier Excel ~46MB, non inclus dans le repo — à télécharger et placer dans `data/`)*

---

## 🖼️ Aperçu

![Vue d'ensemble du dashboard](screenshots/dashboard_overview.png)
*KPIs clés (nombre de clients, valeur moyenne, clients à forte valeur...) et répartition des segments*

![Visualisation 3D interactive des clusters (Recency × Frequency × MonetaryValue)](screenshots/clusters_3d_plotly.png)
*Exploration interactive en 3D des clusters RFM, filtrable par segment*

---

## 🎯 Le défi

Segmenter une large base de transactions e-commerce pour identifier des groupes de clients avec des comportements d'achat distincts, tout en gérant correctement les **outliers** (clients atypiques à très haute ou très basse valeur) qui, non traités, auraient faussé le clustering.

## 🔧 Stack technique

| Composant | Outil |
|---|---|
| Langage | Python |
| Interface | Streamlit |
| Clustering | Scikit-learn (KMeans) |
| Méthode | RFM Analysis (Recency, Frequency, Monetary) |
| Visualisation | Plotly (3D interactif) |

## ⚙️ Démarche

1. **Chargement & exploration** — lecture du fichier Excel, statistiques descriptives, détection des problèmes de qualité de données
2. **Nettoyage des données** — filtrage des transactions valides, suppression des codes produits non-standards, gestion des Customer ID manquants, suppression des prix négatifs/nuls
3. **Feature engineering RFM** :
   - **Recency** : nombre de jours depuis le dernier achat
   - **Frequency** : nombre d'achats
   - **MonetaryValue** : dépense totale (Quantité × Prix)
4. **Gestion des outliers** — séparation des clients atypiques pour un clustering plus robuste
5. **Clustering** — normalisation des variables RFM (StandardScaler) puis application de **K-Means**
6. **Dashboard Streamlit** — visualisation interactive des clusters, métriques par segment, filtrage outliers/non-outliers, export CSV par segment

![Profils détaillés par segment (Recency, Frequency, MonetaryValue)](screenshots/segment_profiles.png)
*Chaque segment dispose de sa propre fiche : nombre de clients, valeur moyenne/totale, et distribution des 3 variables RFM*

## 📌 Segments identifiés

| Cluster | Label | Profil |
|---|---|---|
| 0 | **RETAIN** | Clients réguliers à fidéliser |
| 1 | **RE-ENGAGE** | Clients à réactiver, en perte d'engagement |
| 2 | **NURTURE** | Clients à fort potentiel à développer |
| 3 | **REWARD** | Clients fidèles à récompenser |
| -1 | **PAMPER** *(outlier)* | Clients à très haute valeur, à chouchouter |
| -2 | **UPSELL** *(outlier)* | Clients à potentiel de montée en gamme |
| -3 | **DELIGHT** *(outlier)* | Clients rares mais très engagés |

*(répartition détaillée visible dans le pie chart de l'aperçu ci-dessus)*

## 💡 Résultat & valeur business

Une segmentation RFM + KMeans robuste permet de transformer une base de transactions brute en **leviers marketing actionnables** : cibler les campagnes de rétention, prioriser les efforts sur les clients à fort potentiel, et adapter le discours selon le profil réel de chaque groupe de clients.

> Une segmentation efficace commence par un nettoyage de données solide, suivi d'une compréhension approfondie du domaine métier.

## 🚀 Installation & utilisation

```bash
git clone https://github.com/imane-el-arrach/Customer-segmentation.git
cd Customer-segmentation

# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

> ⚠️ Le dataset n'est pas inclus dans le repo (trop volumineux). Télécharge-le [ici](https://archive.ics.uci.edu/dataset/502/online+retail+ii) et place-le dans le dossier `data/`.

## 📂 Structure du projet

```
Customer-segmentation/
├── data/               # Dataset (à télécharger, non versionné)
├── notebooks/          # Notebook d'exploration et de clustering
├── screenshots/        # Captures utilisées dans ce README
├── app.py              # Application Streamlit
├── requirements.txt
└── README.md
```

## 👩‍💻 Auteure

**Imane El Arrach** — Élève-ingénieure en Génie Informatique, spécialité Ingénierie des Données & IA, ENSA Safi
[LinkedIn](http://www.linkedin.com/in/imane-el-arrach-7a88ab325) · [GitHub](https://github.com/imane-el-arrach)

