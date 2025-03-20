package org.example.classification;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.FileNotFoundException;

public class Word2VecLoader {

    public static INDArray getFeatureVector(String text, Word2Vec word2Vec) {
        String[] words = text.split(" "); // Tokenize message
        INDArray vectorSum = null;

        for (String word : words) {
            if (word2Vec.hasWord(word)) { // Check if the word is in the vocabulary
                INDArray wordVector = word2Vec.getWordVectorMatrix(word);
                if (vectorSum == null) {
                    vectorSum = wordVector.dup(); // Initialize vector
                } else {
                    vectorSum.addi(wordVector); // Add word vector
                }
            }
        }

        return vectorSum.div(words.length); // Return the average vector
    }

    public static void loadWord2Vec() throws FileNotFoundException {
        // Path to the pre-trained Word2Vec model
        File word2VecFile = new ClassPathResource("GoogleNews-vectors-negative300.bin").getFile();

        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(word2VecFile);
        System.out.println("Words similar to 'offer': " + word2Vec.wordsNearest("offer", 10));
    }

    public static void main(String[] args) throws FileNotFoundException {
        Word2VecLoader.loadWord2Vec();
        //INDArray spamVector = getFeatureVector("win a free vacation now", word2Vec);
        //INDArray nonSpamVector = getFeatureVector("hello, how are you?", word2Vec);
        //
        //// Add labels: 1 for spam, 0 for non-spam
        //DataSet dataSet = new DataSet(spamVector, Nd4j.create(new float[] {1}));
    }
}   
