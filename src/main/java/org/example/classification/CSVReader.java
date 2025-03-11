package org.example.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class CSVReader {
    private static int NUM_OF_HEADERS_IN_CSV = 1;

    public static DataSet fetchShuffledCSVData(String fileName, int numOFRows, int featuresCount, int classCount){
        DataSet dataSet = null;
        try(RecordReader recordReader = new CSVRecordReader(NUM_OF_HEADERS_IN_CSV, ',')) {
            FileSplit split = new FileSplit(new ClassPathResource(fileName).getFile());
            System.out.println(split.length());
            System.out.println(split.getRootDir());
            recordReader.initialize(split);
            DataSetIterator dataSetIterator =  new RecordReaderDataSetIterator(recordReader, numOFRows, featuresCount, classCount);
            dataSet = dataSetIterator.next();
            dataSet.shuffle(42);
        } catch (Exception e) {
            throw new RuntimeException(String.format("Unable to read data from (%s)", fileName), e);
        }
        return dataSet;
    }
}
