# Installation

Cloner ce dépot git :

`git clone https://github.com/Kaduo/mira-inauguration`

Aller sur la branche `big-cleanup` (expérimentale mais normalement plus simple d'utilisation) :

`bash git checkout big-cleanup`

Créer un environnement virtuel python :

`python -m venv .venv`

L'activer (cette commande fonctionne pour bash/zsh, pas sous Windows) :

`source .venv/bin/activate`

Installer les dépendances :

`pip install .`

# Utilisation

La configuration se fait à l'aide du fichier `config.toml`. Le script `main.py` permet de faire dessiner le robot

## Configuration

- `above_origin`, `above_p1`

Dans le fichier `config.toml`, changer les valeurs "above_origin", "above_p1" et "above_p2" à des points qui sont "au dessus" des trois points qui forment le repère sur lequel le robot va dessiner. Le point "above_origin" doit être à l'angle du repère.

# Limitations

Pour le moment, ne fonctionne pas avec des images au format carré.