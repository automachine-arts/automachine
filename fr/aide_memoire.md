# Commandes pratiques pour le Terminal

  `<TAB>` : permet l'autocomplétion d'un mot que l'on a commencé à écrire. par exemple, si on souhaite se déplacer
  vers un nouveau dossier appelé "automachine" et que l'on tape "cd auto<TAB>", on obtiendra "cd automachine" de façon
  automatique.
  
  `<UP ARROW>` : appuyer sur la flèche vers le haut permet d'afficher la ou les commandes précédemment entré es.
  
  ## Windows

    `dir` : affiche dans le Terminal la liste des fichiers dans le dossier actif

    `cd <FOLDER NAME>` : permet d'accéder au dossier se nommant <FOLDER NAME>


  ## Linux/Mac OSX

    `ls` : affiche dans le Terminal la liste des fichiers dans le dossier actif

    `cd` <FOLDER NAME> : permet d'accéder au dossier se nommant <FOLDER NAME>

    `sh` <SCRIPT>.sh : exécute le script <SCRIPT>.sh


# Commandes GIT

  `git checkout <URL>` : fait une copie locale d'un dossier contenant du code source situé sur un serveur.
  
  `git pull` : permet de mettre à jour la copie locale avec les changements situés sur le serveur.

  `git diff` : permet de visualiser les changements effectués localement.

# Commandes CONDA

  `conda create <ENVIRONMENT>` : crée un nouvel environnement

  `conda install <PACKAGE> -c <HOST>` : lance l'installation du paquet <PACKAGE> afin de pouvoir l'inclure dans un code 
  Python. il est possible qu'un hôte <HOST> (endroit à partir duquel installer les fichiers) soit précisé, mais c'est optionnel.
  
  `conda activate <ENVIRONMENT>` : permet de lancer et d'accéder à l'environnement <ENVIRONMENT>, qui aura sa propre
  installation de Python (par exemple, 3.7, 3.8...) ainsi que son propre ensemble de paquets disponibles afin d'être
  utilisés dans notre code Python.
  
# Apprentissage machine - définitions et concepts

  Nombre à virgule flottante (loating point number): un nombre réel avec un point, tel qu'approximé par un ordinateur.
    Note: essayez d'ajouter 3 fois le nombre 0.1 dans un interprète Python - vous serez probablement surpris.e !

  Tenseur : Un tenseur est une matrice multi-dimensionnelle; par exemple, une matrice cubique dans l'espace, où chaque
  point situé sur une combinaison ligne/colonne/profondeur contient un nombre à virgule flottante.

  Apprentissage : Apprendre les paramètres numériques (les valeurs des neurones) qui permettront d'obtenir la réponse
    correcte, dépendamment des données observées.

  Époque : une étape d'apprentissage, qui calcule les différences à apporter au modèle afin d'être plus près du résultat voulu.

  Prévision: obtenue en utilisant un modèle ayant complété sa phase d'apprentissage, sur des nouvelles données, afin de générer une prévision.

  Données de test: après la période d'apprentissage, nous testons le modèle sur des données que celui-ci n'a jamais vu, afin de vérifier que le modèle fonctionne.

  Données de validation: données sur lesquelles il est possible de faire l'apprentissage, mais que l'on utilise pour tester à la place.

# Concepts Python

  Commentaires: tout ce qui suit un symbole '#' est destiné à la lecture humaine, et non pour la lecture machine.

  ## Variables

    # 1. crée une nouvelle variable se nommant 'x' et ayant une valeur de 3.  

    x = 3 

    # 2. crée une nouvelle variable se nommant 'x' qui est un tensur (matrice de dimension 3 ou plus)
    
    x = Tensor() 
  
  ## Fonctions

    # 1. crée une nouvelle fonction prenant deux arguments, 'mes' et 'variables', qui retourne le résultat de l'addition de 3 et de 5.

    def mafonction(mes, variables): 
      x = 3 + 5
      return x
    
    # 2. prend l'objet 'x' et appelle sa fonction 'train' en lui donnant l'argument 'data'.
    
    x.train(data) 

  ## Imports
          
      from <PACKAGE> import <fUNCTION OR OBJECT> 
  
      # est déclaré au début du fichier afin que l'interprète Python puisse trouver le code nécessaire à l'exécution du programme.

  ## Indexation
  
      data[début:fin]
      
      # sélectionne des éléments d'une liste, d'un tenseur, d'un tableau ou d'une autre structure de données à partir de 'début' mais sans inclure 'fin'.
