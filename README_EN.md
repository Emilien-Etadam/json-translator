# 🌍 JSON Selective Translator with Ollama

A Python tool with Gradio interface for selectively translating specific JSON keys using local Ollama models.

**[Français](README.md)** | **English**

## ✨ Features

### Main Features
- 🎯 **Selective Translation**: Translate only the JSON keys you specify
- 🤖 **Local Ollama**: Uses local AI models (no external API required)
- 🎨 **Gradio Interface**: Intuitive and modern web interface
- 🔄 **Nested Path Support**: `user.profile.description`, `items[0].title`
- 🎭 **Regex Patterns**: `*.description` for all "description" keys
- 💾 **Smart Caching**: Avoids redundant translations
- 🔒 **Variable Preservation**: Protects `{{var}}`, `{var}`, `%s`, etc.

### Advanced Features
- 👁️ **Dry-run mode**: Preview keys that will be translated
- 📊 **Progress Bar**: Real-time tracking
- 📋 **Log Export**: Complete translation history
- ✅ **JSON Validation**: Automatic validity checking
- 🎯 **Batch Translation**: Optimized for large structures

## 🚀 Installation

### Deployment

**Option 1: Deploy on Coolify** (recommended for servers)
- See complete guide: [DEPLOY_COOLIFY.md](DEPLOY_COOLIFY.md)
- Native Nixpacks support
- Simple configuration via environment variables

**Option 2: Local installation**

### Prerequisites
1. **Python 3.8+**
2. **Ollama** installed and running

#### Installing Ollama
```bash
# Windows, macOS, Linux
# Download from https://ollama.ai

# Then install a model (example with llama2)
ollama pull llama2

# Or with mistral (recommended for French)
ollama pull mistral

# Or with mixtral for better performance
ollama pull mixtral
```

### Script Installation
```bash
# Clone or download the files
cd json_translator

# Install Python dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Launch
```bash
python json_translator.py
```

The Gradio interface will automatically open in your browser at `http://127.0.0.1:7860`

### Step-by-Step Guide

1. **Load your JSON file**
   - Click "Upload JSON File"
   - Select your `.json` file
   - The JSON appears in the "Source JSON" tab

2. **Configure translation**
   - Select the **Ollama model** (e.g., `mistral`, `llama2`)
   - Choose the **target language** (French, English, Spanish, etc.)

3. **Specify keys to translate**
   ```
   title
   description
   user.profile.bio
   items[0].content
   *.text
   sections.*.description
   ```

4. **Options**
   - ☑️ Check "Dry Run" to preview without translating
   - 🚀 Click "Translate"

5. **View results**
   - "Translated JSON" tab: Translated JSON
   - "Selected Keys" tab: List of processed keys
   - Status: Translation information

6. **Download results**
   - Click "Download Translated JSON"
   - Export logs with "Export Translation Log"

## 🎯 Key Pattern Examples

### Simple Patterns
```
title                 # All "title" keys at any level
description          # All "description" keys
author               # All "author" keys
```

### Nested Paths
```
user.name            # "name" in the "user" object
user.profile.bio     # "bio" in "user.profile"
config.app.title     # "title" in "config.app"
```

### Arrays
```
items[0].title       # "title" of the first "items" element
users[2].description # "description" of the third user
```

### Wildcards (*)
```
*.description        # All "description" keys at any level
user.*.name          # All "name" keys under "user"
sections.*.title     # All "title" in "sections"
**.content           # "content" at any depth
```

### Advanced Regex
```
^(title|name)$       # "title" OR "name" keys
.*_text$             # Keys ending with "_text"
```

## 📝 JSON Example

Create an `example.json` file:

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

**Keys to translate:**
```
title
description
author.bio
sections.*.heading
sections.*.content
```

**Result**: Texts are translated, but:
- `author.name` remains "John Doe" (not specified)
- `config.version` remains "1.0.0" (not specified)
- Variables `{{topic}}`, `%s`, `{count}` are preserved
- Metadata and IDs remain intact

## ⚙️ Advanced Configuration

### Automatically Preserved Variables
The script automatically detects and preserves:
- `{{variable}}` - Double braces
- `{variable}` - Single braces
- `%(variable)s`, `%(name)d` - Python format
- `%s`, `%d` - Printf style
- `${variable}` - Shell style
- `[[variable]]` - Double brackets

### Translation Cache
- Identical translations are cached
- Reduces translation time and API calls
- Use "Clear Cache" to empty the cache

### Recommended Ollama Models

**For French:**
- `mistral` - Excellent for French (7B parameters)
- `mixtral` - Very high quality (47B parameters, slower)
- `llama2` - Good balance (7B/13B/70B)

**For other languages:**
- `gemma` - Google multilingual
- `vicuna` - Fine-tuned for conversations

```bash
# Install multiple models
ollama pull mistral
ollama pull mixtral
ollama pull llama2
```

## 🔧 Troubleshooting

### "No Ollama models found"
```bash
# Check that Ollama is running
ollama list

# Install a model if necessary
ollama pull mistral
```

### "Cannot connect to Ollama"
```bash
# Check the Ollama service
# Windows: Look for "Ollama" in services
# Linux/Mac:
ollama serve
```

### Memory Error
```bash
# Use a smaller model
ollama pull mistral:7b-instruct

# Or free up memory
ollama rm <model-name>
```

### Poor Translation Quality
1. Try another model (mistral or mixtral recommended)
2. Check that specified keys are correct
3. Use dry-run mode to verify selected keys

## 📊 Statistics and Logs

### During Translation
- Progress bar with number of keys processed
- Display of text being translated
- Cache statistics (hits/misses)

### After Translation
- Total number of translations performed
- Number of modified keys
- Cache hits/misses ratio

### Log Export
Click "Export Translation Log" to generate a text file containing:
- All translations performed
- Timestamp of each translation
- Target language used
- Source and translated text
- Cache statistics

## 🏗️ Code Architecture

```
json_translator.py
├── TranslationCache        # In-memory cache with MD5 hash
├── VariablePreserver       # Variable detection and protection
├── JSONKeySelector         # Key selection with patterns/regex
├── OllamaTranslator       # Main translation engine
└── create_gradio_interface # Gradio user interface
```

### Main Classes

**TranslationCache**
- In-memory translation storage
- Key: hash(text + language + model)
- Hits/misses statistics

**VariablePreserver**
- Detects variable patterns
- Replaces them with placeholders during translation
- Restores them after translation

**JSONKeySelector**
- Flattens JSON to dot notation
- Matches patterns (wildcards, regex)
- Reconstructs JSON with translated values

**OllamaTranslator**
- Calls to local Ollama API
- Cache and variable management
- Batch translation
- Log export

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Propose new features
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) - Local AI models
- [Gradio](https://gradio.app) - Python web interface
- Open source community

## 📞 Support

For questions or help:
1. Check the Troubleshooting section above
2. Verify all dependencies are installed
3. Ensure Ollama is working properly

---

**Good to know:**
- ✅ Works 100% locally (no internet connection required)
- ✅ No usage limits
- ✅ Complete data privacy
- ✅ Multilingual support
- ✅ Free and open source
