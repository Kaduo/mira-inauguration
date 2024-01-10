# Installation

Cloner ce dépot git :

```bash
git clone https://github.com/Kaduo/mira-inauguration
```

Aller sur la branche `big_cleanup` (expérimentale mais normalement plus simple d'utilisation) :

```bash 
git checkout big_cleanup
```

Créer un environnement virtuel python :

```bash
python -m venv .venv
```

L'activer (cette commande fonctionne pour bash/zsh, pas sous Windows) :

```bash
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install .
```

# Utilisation

La configuration se fait à l'aide du fichier `config.toml`. Le script `main.py` permet de faire dessiner le robot, soit à partir d'une image enregistrée, soit en mode "photomaton". Le raccourci Ctrl-C permet de mettre en pause le dessin, il faut alors taper `continue` dans la ligne de commande pour le faire reprendre. Pour arrêter le dessin, utiliser le raccourci Ctrl-Z.

## Configuration

Les principales options sont :

### Général

- `enable_photomaton` : en mode photomaton, le programme prend une photo avec la webcam de l'ordinateur.
- `image_path` : le chemin de l'image à faire dessiner par le robot (fonctionne uniquement si `enable_photomaton` est mis à false)
- `robot_ip` : l'adresse ip du robot

### Calibration

- `above_origin`, `above_p1`, `above_p2` : ces points doivent être situés légèrement au dessus des points qui constituent le repère dans lequel va dessiner le robot. Par exemple, si l'origine de ce repère est en (4, 72, 56), `above_origin` doit être configuré à [4, 72, 60].
- `epsilon` : plus ce nombre est élevé, plus le robot va appuyer fort sur la feuille lors de la calibration. Un epsilon trop grand et le robot risque de casser le stylo, un epsilon trop bas et il risque de ne pas toucher la feuille du tout.
- `relative-epsilon` : pas encore implémenté

### Photomaton

- `camera_index` : l'id de la caméra à utiliser pour le photomaton

# Limitations

Pour le moment, ne fonctionne pas avec des images au format carré.

Toutes les options présentes dans `config.toml` ne sont pas encore implémentées.
