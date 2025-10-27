# 🌍 JSON Selective Translator with Ollama

Un outil Python avec interface Gradio pour traduire sélectivement des clés JSON spécifiques en utilisant des modèles Ollama locaux.

## ✨ Fonctionnalités

### Principales
- 🎯 **Traduction sélective** : Traduisez uniquement les clés JSON que vous spécifiez
- 🤖 **Ollama local** : Utilise des modèles d'IA locaux (aucune API externe requise)
- 🎨 **Interface Gradio** : Interface web intuitive et moderne
- 🔄 **Support des chemins imbriqués** : `user.profile.description`, `items[0].title`
- 🎭 **Patterns regex** : `*.description` pour toutes les clés "description"
- 💾 **Cache intelligent** : Évite les traductions redondantes
- 🔒 **Préservation des variables** : Protège `{{var}}`, `{var}`, `%s`, etc.

### Avancées
- 👁️ **Mode dry-run** : Prévisualisez les clés qui seront traduites
- 📊 **Barre de progression** : Suivi en temps réel
- 📋 **Export de logs** : Historique complet des traductions
- ✅ **Validation JSON** : Vérification automatique de la validité
- 🎯 **Traduction par lots** : Optimisation pour les grandes structures

## 🚀 Installation

### Prérequis
1. **Python 3.8+**
2. **Ollama** installé et en cours d'exécution

#### Installation d'Ollama
```bash
# Windows, macOS, Linux
# Téléchargez depuis https://ollama.ai

# Puis installez un modèle (exemple avec llama2)
ollama pull llama2

# Ou avec mistral (recommandé pour le français)
ollama pull mistral

# Ou avec mixtral pour de meilleures performances
ollama pull mixtral
```

### Installation du script
```bash
# Clonez ou téléchargez les fichiers
cd "Markdown Translate"

# Installez les dépendances Python
pip install -r requirements.txt
```

## 📖 Utilisation

### Lancement
```bash
python json_translator.py
```

L'interface Gradio s'ouvrira automatiquement dans votre navigateur à l'adresse `http://127.0.0.1:7860`

### Guide pas à pas

1. **Chargez votre fichier JSON**
   - Cliquez sur "Upload JSON File"
   - Sélectionnez votre fichier `.json`
   - Le JSON s'affiche dans l'onglet "Source JSON"

2. **Configurez la traduction**
   - Sélectionnez le **modèle Ollama** (ex: `mistral`, `llama2`)
   - Choisissez la **langue cible** (français, anglais, espagnol, etc.)

3. **Spécifiez les clés à traduire**
   ```
   title
   description
   user.profile.bio
   items[0].content
   *.text
   sections.*.description
   ```

4. **Options**
   - ☑️ Cochez "Dry Run" pour prévisualiser sans traduire
   - 🚀 Cliquez sur "Translate"

5. **Consultez les résultats**
   - Onglet "Translated JSON" : JSON traduit
   - Onglet "Selected Keys" : Liste des clés traitées
   - Statut : Informations sur la traduction

6. **Téléchargez le résultat**
   - Cliquez sur "Download Translated JSON"
   - Exportez les logs avec "Export Translation Log"

## 🎯 Exemples de patterns de clés

### Patterns simples
```
title                 # Toutes les clés "title" à n'importe quel niveau
description          # Toutes les clés "description"
author               # Toutes les clés "author"
```

### Chemins imbriqués
```
user.name            # "name" dans l'objet "user"
user.profile.bio     # "bio" dans "user.profile"
config.app.title     # "title" dans "config.app"
```

### Tableaux
```
items[0].title       # "title" du premier élément de "items"
users[2].description # "description" du troisième utilisateur
```

### Wildcards (*)
```
*.description        # Toutes les clés "description" à n'importe quel niveau
user.*.name          # Toutes les clés "name" sous "user"
sections.*.title     # Tous les "title" dans "sections"
**.content           # "content" à n'importe quelle profondeur
```

### Regex avancé
```
^(title|name)$       # Clés "title" OU "name"
.*_text$             # Clés se terminant par "_text"
```

## 📝 Exemple JSON

Créez un fichier `example.json` :

```json
{
  "title": "Welcome to my website",
  "description": "This is a great website about {{topic}}",
  "author": {
    "name": "John Doe",
    "bio": "Software developer passionate about %s"
  },
  "sections": [
    {
      "heading": "About Us",
      "content": "We are a team of {count} developers",
      "metadata": {
        "created": "2024-01-01",
        "id": 123
      }
    },
    {
      "heading": "Contact",
      "content": "Reach us at contact@example.com"
    }
  ],
  "config": {
    "version": "1.0.0",
    "debug": false
  }
}
```

**Clés à traduire :**
```
title
description
author.bio
sections.*.heading
sections.*.content
```

**Résultat** : Les textes sont traduits, mais :
- `author.name` reste "John Doe" (non spécifié)
- `config.version` reste "1.0.0" (non spécifié)
- Les variables `{{topic}}`, `%s`, `{count}` sont préservées
- Les métadonnées et IDs restent intacts

## ⚙️ Configuration avancée

### Variables préservées automatiquement
Le script détecte et préserve automatiquement :
- `{{variable}}` - Double accolades
- `{variable}` - Accolades simples
- `%(variable)s`, `%(name)d` - Python format
- `%s`, `%d` - Printf style
- `${variable}` - Shell style
- `[[variable]]` - Double crochets

### Cache de traduction
- Les traductions identiques sont mises en cache
- Réduit le temps de traduction et les appels API
- Utilisez "Clear Cache" pour vider le cache

### Modèles Ollama recommandés

**Pour le français :**
- `mistral` - Excellent pour le français (7B paramètres)
- `mixtral` - Très haute qualité (47B paramètres, plus lent)
- `llama2` - Bon équilibre (7B/13B/70B)

**Pour d'autres langues :**
- `gemma` - Multilingue Google
- `vicuna` - Fine-tuné pour conversations

```bash
# Installer plusieurs modèles
ollama pull mistral
ollama pull mixtral
ollama pull llama2
```

## 🔧 Dépannage

### "No Ollama models found"
```bash
# Vérifiez qu'Ollama est en cours d'exécution
ollama list

# Installez un modèle si nécessaire
ollama pull mistral
```

### "Cannot connect to Ollama"
```bash
# Vérifiez le service Ollama
# Windows : Cherchez "Ollama" dans les services
# Linux/Mac :
ollama serve
```

### Erreur de mémoire
```bash
# Utilisez un modèle plus petit
ollama pull mistral:7b-instruct

# Ou libérez de la mémoire
ollama rm <model-name>
```

### Traductions de mauvaise qualité
1. Essayez un autre modèle (mistral ou mixtral recommandés)
2. Vérifiez que les clés spécifiées sont correctes
3. Utilisez le mode dry-run pour vérifier les clés sélectionnées

## 📊 Statistiques et logs

### Pendant la traduction
- Barre de progression avec nombre de clés traitées
- Affichage du texte en cours de traduction
- Statistiques de cache (hits/misses)

### Après la traduction
- Nombre total de traductions effectuées
- Nombre de clés modifiées
- Ratio cache hits/misses

### Export de logs
Cliquez sur "Export Translation Log" pour générer un fichier texte contenant :
- Toutes les traductions effectuées
- Horodatage de chaque traduction
- Langue cible utilisée
- Texte source et traduit
- Statistiques du cache

## 🏗️ Architecture du code

```
json_translator.py
├── TranslationCache        # Cache en mémoire avec hash MD5
├── VariablePreserver       # Détection et protection des variables
├── JSONKeySelector         # Sélection des clés avec patterns/regex
├── OllamaTranslator       # Moteur de traduction principal
└── create_gradio_interface # Interface utilisateur Gradio
```

### Classes principales

**TranslationCache**
- Stockage des traductions en mémoire
- Clé : hash(texte + langue + modèle)
- Statistiques hits/misses

**VariablePreserver**
- Détecte les patterns de variables
- Les remplace par des placeholders pendant la traduction
- Les restaure après traduction

**JSONKeySelector**
- Flatten le JSON en notation point
- Match les patterns (wildcards, regex)
- Reconstruction du JSON avec valeurs traduites

**OllamaTranslator**
- Appels à l'API Ollama locale
- Gestion du cache et des variables
- Traduction par lots
- Export de logs

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer de nouvelles fonctionnalités
- Améliorer la documentation

## 📄 Licence

Ce projet est libre d'utilisation pour des projets personnels et commerciaux.

## 🙏 Remerciements

- [Ollama](https://ollama.ai) - Modèles d'IA locaux
- [Gradio](https://gradio.app) - Interface web Python
- Communauté open source

## 📞 Support

Pour des questions ou de l'aide :
1. Consultez la section Dépannage ci-dessus
2. Vérifiez que toutes les dépendances sont installées
3. Assurez-vous qu'Ollama fonctionne correctement

---

**Bon à savoir :**
- ✅ Fonctionne 100% en local (aucune connexion internet requise)
- ✅ Pas de limite d'utilisation
- ✅ Confidentialité totale des données
- ✅ Support multilingue
- ✅ Gratuit et open source
