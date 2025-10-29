# 🚀 Déploiement sur Coolify avec Nixpacks

Guide pour déployer JSON Translator sur Coolify en utilisant Nixpacks.

## 📋 Prérequis

- Serveur Coolify configuré et accessible
- Ollama installé quelque part (sur votre PC ou sur le serveur)
- Git repository accessible (GitHub, GitLab, etc.)

---

## 🔧 Configuration Coolify

### 1. Créer une nouvelle application

Dans Coolify :
1. Cliquez sur **"New Resource"** → **"Application"**
2. Sélectionnez votre repository Git
3. Sélectionnez la branche (ex: `main` ou `claude/check-github-readiness-...`)
4. **Build Pack** : Sélectionnez **"Nixpacks"** (détection automatique)

### 2. Variables d'environnement

Dans l'onglet **Environment Variables**, ajoutez :

#### **Option A : Ollama sur votre PC local**

```env
# Requis
OLLAMA_HOST=http://192.168.1.100:11434
PORT=7860
ENVIRONMENT=production

# Optionnel
HOST=0.0.0.0
```

⚠️ **Remplacez `192.168.1.100`** par l'IP locale de votre PC où Ollama tourne.

**Trouver votre IP locale** :
```bash
# Sur votre PC avec Ollama
ip addr show | grep "inet " | grep -v 127.0.0.1
# ou
hostname -I
```

**Vérifier qu'Ollama est accessible** depuis le serveur Coolify :
```bash
# Depuis le serveur Coolify
curl http://192.168.1.100:11434/api/tags
```

#### **Option B : Ollama sur le serveur Coolify**

Si vous installez Ollama directement sur le serveur Coolify (NixOS) :

```env
# Requis
OLLAMA_HOST=http://localhost:11434
PORT=7860
ENVIRONMENT=production
```

**Installation Ollama sur NixOS** :
```nix
# /etc/nixos/configuration.nix
{
  services.ollama = {
    enable = true;
    host = "0.0.0.0";
    port = 11434;
  };
}
```

Puis :
```bash
sudo nixos-rebuild switch
ollama pull mistral  # ou llama2
```

### 3. Configuration réseau (Coolify)

1. **Port** : `7860` (ou autre selon votre PORT)
2. **Protocol** : `HTTP`
3. **Domaine** : Configurez un domaine ou utilisez l'IP:port

---

## 🚀 Déploiement

### Premier déploiement

1. Dans Coolify, cliquez sur **"Deploy"**
2. Nixpacks va automatiquement :
   - Détecter Python 3.11
   - Installer `requirements.txt`
   - Lancer `python json_translator.py`
3. Attendez que le build se termine (~2-5 minutes)
4. L'application sera accessible sur `http://votre-serveur:7860`

### Déploiements suivants

À chaque push sur la branche configurée, Coolify redéploie automatiquement (si activé).

---

## ✅ Vérification

### 1. Vérifier les logs

Dans Coolify → Votre application → **Logs** :

```
INFO - Ollama client configured to connect to: http://192.168.1.100:11434
INFO - Launching Gradio interface on 0.0.0.0:7860
INFO - Starting JSON Selective Translator with Ollama
Running on local URL:  http://0.0.0.0:7860
```

### 2. Tester la connexion Ollama

```bash
# Depuis votre navigateur
http://votre-serveur:7860

# Dans l'interface, vérifiez que les modèles Ollama apparaissent
```

Si vous voyez la liste des modèles → ✅ Connexion Ollama OK

### 3. Tester une traduction

1. Uploadez un fichier JSON de test
2. Sélectionnez des clés
3. Choisissez une langue
4. Cliquez "Translate"

---

## 🔧 Dépannage

### Problème : "Cannot connect to Ollama"

**Cause** : Coolify ne peut pas atteindre Ollama

**Solutions** :

1. **Vérifier la connexion réseau** :
   ```bash
   # Depuis le serveur Coolify
   curl http://192.168.1.100:11434/api/tags
   ```

2. **Vérifier le firewall** sur votre PC :
   ```bash
   # Permettre le port 11434
   sudo ufw allow 11434
   # ou
   sudo firewall-cmd --add-port=11434/tcp --permanent
   ```

3. **Vérifier qu'Ollama écoute sur toutes les interfaces** :
   ```bash
   # Sur votre PC
   ss -tlnp | grep 11434
   # Doit afficher 0.0.0.0:11434 (pas 127.0.0.1:11434)
   ```

   Si Ollama écoute sur 127.0.0.1 uniquement :
   ```bash
   # Arrêter Ollama
   killall ollama

   # Redémarrer avec OLLAMA_HOST
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```

### Problème : "No models found"

**Cause** : Aucun modèle Ollama installé

**Solution** :
```bash
# Sur la machine avec Ollama
ollama pull mistral
ollama pull llama2
ollama list  # Vérifier
```

### Problème : Port déjà utilisé

**Cause** : Le port 7860 est déjà pris

**Solution** : Changer la variable `PORT` dans Coolify :
```env
PORT=8080
```

### Problème : Application inaccessible

**Cause** : L'app écoute sur 127.0.0.1 au lieu de 0.0.0.0

**Solution** : Vérifier que `HOST=0.0.0.0` est configuré dans les variables d'environnement.

---

## 📊 Architecture réseau

### Option A : Ollama distant
```
Internet
   ↓
Coolify Serveur (192.168.1.50)
   └── JSON Translator (Port 7860)
       ↓ HTTP
Votre PC (192.168.1.100)
   └── Ollama (Port 11434)
```

### Option B : Tout sur Coolify
```
Internet
   ↓
Coolify Serveur (192.168.1.50)
   ├── JSON Translator (Port 7860)
   └── Ollama (Port 11434)
       └── Communication localhost
```

---

## 🔒 Sécurité

### Si vous utilisez Ollama distant (Option A)

⚠️ **Ollama n'a pas d'authentification** par défaut

**Recommandations** :

1. **Firewall** : Limitez l'accès au port 11434 :
   ```bash
   # Autoriser uniquement le serveur Coolify
   sudo ufw allow from 192.168.1.50 to any port 11434
   ```

2. **VPN/Tailscale** : Utilisez un réseau privé virtuel

3. **Reverse Proxy** : Ajoutez une authentification avec Nginx/Caddy

---

## 💡 Optimisations

### 1. Activer le déploiement automatique

Dans Coolify → Votre application → **Settings** :
- Cochez "Automatic Deployment"
- Coolify redéploiera à chaque push

### 2. Configurer un domaine

Dans Coolify → Votre application → **Domains** :
```
json-translator.votre-domaine.com
```

Coolify génère automatiquement les certificats SSL (Let's Encrypt).

### 3. Logs persistants

Les logs Gradio sont affichés dans Coolify. Pour les logs structurés JSON, vous pouvez les exporter :

```bash
# Dans Coolify, accédez au shell du container
# Puis :
cat /app/logs/*.log | grep "translation_completed" | jq .
```

---

## 📚 Ressources

- [Documentation Nixpacks](https://nixpacks.com/docs)
- [Documentation Coolify](https://coolify.io/docs)
- [Documentation Ollama](https://ollama.ai)
- [Documentation Gradio](https://gradio.app)

---

## 🆘 Support

Si vous rencontrez des problèmes :

1. Vérifiez les logs dans Coolify
2. Testez la connexion Ollama manuellement
3. Vérifiez les variables d'environnement
4. Consultez ce guide de dépannage

---

**Bon déploiement !** 🚀
