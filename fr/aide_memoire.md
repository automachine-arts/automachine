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

## Commentaires
  
  Tout ce qui suit un symbole '#' est destiné à la lecture humaine, et non pour la lecture machine.

## Variables

  Créer une nouvelle variable se nommant 'x' et ayant une valeur de 3:

    x = 3 

  Créer une nouvelle variable se nommant 'x' qui est un tenseur (matrice de dimension 3 ou plus)
    
    x = Tensor() 
  
## Fonctions

  Crée une nouvelle fonction prenant deux arguments, 'mes' et 'variables', qui retourne le résultat de l'addition de 3 et de 5.

    def mafonction(mes, variables): 
      x = 3 + 5
      return x
    
  Prend l'objet 'x' et appelle sa fonction 'train' en lui donnant l'argument 'data'.
    
    x.train(data) 

## Imports
          
      from <PACKAGE> import <fUNCTION OR OBJECT> 
  
  Une importation est déclarée au début du fichier afin que l'interprète Python puisse trouver le code nécessaire à l'exécution du programme.

## Indexation
  
      data[début:fin]
      
  L'indexaction permet de sélectionner des éléments d'une liste, d'un tenseur, d'un tableau ou d'une autre structure de données à partir de 'début' mais sans inclure 'fin'.
