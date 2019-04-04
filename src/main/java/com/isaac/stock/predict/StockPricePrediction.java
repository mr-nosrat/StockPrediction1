package com.isaac.stock.predict;

import com.isaac.stock.model.RecurrentNets;
import com.isaac.stock.representation.PriceCategory;
import com.isaac.stock.representation.StockData;
import com.isaac.stock.representation.StockDataSetIterator;
import com.isaac.stock.utils.PlotUtil;
import javafx.util.Pair;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.cublas;
import org.bytedeco.javacpp.cuda;
import org.bytedeco.javacpp.cusparse;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Created by zhanghao on 26/7/17.
 * Modified by zhanghao on 28/9/17.
 *
 * @author ZHANG HAO
 */
public class StockPricePrediction {

    private static final Logger log = LoggerFactory.getLogger(StockPricePrediction.class);

    private static int exampleLength = 22; // time series length, assume 22 working days per month

    public static void main(String[] args) throws IOException {
//        try {
//            Loader.load(cusparse.class);
//        } catch (UnsatisfiedLinkError e) {
//            String path = Loader.cacheResource(cuda.class, "windows-x86_64/jnicusparse.dll").getPath();
//            try {
//                new ProcessBuilder("C:/Users/nosrat/Downloads/Dependencies_x64_Release/DependenciesGui.exe", path).start().waitFor();
//            } catch (InterruptedException e1) {
//                e1.printStackTrace();
//            }
//        }
        String file = new ClassPathResource("Irka.Part.csv").getFile().getAbsolutePath();
        String symbol = "Irka.Part"; // stock name
        int batchSize = 64; // mini-batch size
        double splitRatio = 0.9; // 90% for training, 10% for testing
        int epochs = 100; // training epochs

        log.info("Create dataSet iterator...");
        PriceCategory category = PriceCategory.CLOSE; // CLOSE: predict close price
        StockDataSetIterator iterator = new StockDataSetIterator(file, symbol, batchSize, exampleLength, splitRatio, category);
        log.info("Load test dataset...");
        List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        log.info("Build lstm networks...");
        MultiLayerNetwork net = RecurrentNets.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());

        log.info("Training...");
        for (int i = 0; i < epochs; i++) {
            while (iterator.hasNext()) net.fit(iterator.next()); // fit model using mini-batch data
            iterator.reset(); // reset iterator
            net.rnnClearPreviousState(); // clear previous state

            if((epochs % 10)==0) {
                INDArray max = Nd4j.create(iterator.getMaxArray());
                INDArray min = Nd4j.create(iterator.getMinArray());
                predictPriceOneAhead(net, test, max, min, category, iterator.getTest1());
            }
        }

        log.info("Saving model...");
        File locationToSave = new File("src/main/resources/StockPriceLSTM_".concat(String.valueOf(category)).concat(".zip"));
        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, true);

        log.info("Load model...");
        net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        log.info("Testing...");
        if (category.equals(PriceCategory.ALL)) {
            INDArray max = Nd4j.create(iterator.getMaxArray());
            INDArray min = Nd4j.create(iterator.getMinArray());
            predictAllCategories(net, test, max, min);
        } else {
//            double max = iterator.getMaxNum(category);
//            double min = iterator.getMinNum(category);
            INDArray max = Nd4j.create(iterator.getMaxArray());
            INDArray min = Nd4j.create(iterator.getMinArray());
            predictPriceOneAhead(net, test, max, min, category, iterator.getTest1());
        }
        log.info("Done...");
    }

    static boolean addNextDay = false;

    private static void predictPriceOneAhead (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double max, double min, PriceCategory category) {
        double[] predicts = new double[testData.size()];
        double[] actuals = new double[testData.size()];
        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1) * (max - min) + min;
            actuals[i] = testData.get(i).getValue().getDouble(0);
        }
        log.info("Print out Predictions and Actual Values...");
        log.info("Predict,Actual");
        for (int i = 0; i < predicts.length; i++) log.info(predicts[i] + "," + actuals[i]);
        log.info("Plot...");
        PlotUtil.plot(predicts, actuals, String.valueOf(category));
    }
    /**
     * Predict one feature of a stock one-day ahead
     */
    private static void predictPriceOneAhead(MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, INDArray max, INDArray min, PriceCategory category, List<StockData> test1) {
        int araySize = testData.size();
        if (addNextDay == false)
            araySize = testData.size() + 1;
        double[] predicts = new double[araySize];
        double[] actuals = new double[araySize];

        for (int i = 0; i < predicts.length; i++) {
            if (i == predicts.length - 1) {
                if (addNextDay)
                    testData.remove(testData.size() - 1);
                INDArray input = Nd4j.create(new int[]{exampleLength, 5}, 'f');
                int r = i + 1;
                StockData stock=null;
                for (int j = r; j < test1.size(); j++) {
                    stock = test1.get(j);
                    input.putScalar(new int[]{j - r, 0}, (stock.getOpen() - min.getDouble(0)) / (max.getDouble(0) - min.getDouble(0)));
                    input.putScalar(new int[]{j - r, 1}, (stock.getClose() - min.getDouble(1)) / (max.getDouble(1) - min.getDouble(1)));
                    input.putScalar(new int[]{j - r, 2}, (stock.getLow() - min.getDouble(2)) / (max.getDouble(2) - min.getDouble(2)));
                    input.putScalar(new int[]{j - r, 3}, (stock.getHigh() - min.getDouble(3)) / (max.getDouble(3) - min.getDouble(3)));
                    input.putScalar(new int[]{j - r, 4}, (stock.getVolume() - min.getDouble(4)) / (max.getDouble(4) - min.getDouble(4)));
                    log.info((j - r) + stock.getDate() + " - " + String.valueOf(stock.getClose()));
                }
                int pil=i;
//                if(!addNextDay)
                    pil=i-1;
                stock.setVolume(4470147);
                stock.setOpen(1971);
                stock.setLow(1950);
                stock.setHigh(2043);
                input.putScalar(new int[]{21, 0}, (stock.getOpen() - min.getDouble(0)) / (max.getDouble(0) - min.getDouble(0)));
                input.putScalar(new int[]{21, 1}, (predicts[pil] - min.getDouble(1)) / (max.getDouble(1) - min.getDouble(1)));
                input.putScalar(new int[]{21, 2}, (stock.getLow() - min.getDouble(2)) / (max.getDouble(2) - min.getDouble(2)));
                input.putScalar(new int[]{21, 3}, (stock.getHigh() - min.getDouble(3)) / (max.getDouble(3) - min.getDouble(3)));
                input.putScalar(new int[]{21, 4}, (stock.getVolume() - min.getDouble(4)) / (max.getDouble(4) - min.getDouble(4)));
                log.info((21) + "Next Day" + " - " + String.valueOf(predicts[pil]));
                INDArray label = Nd4j.create(new int[]{1}, 'f');
                label.putScalar(new int[]{0}, predicts[pil]);
//            if (category.equals(PriceCategory.ALL)) {
//                label = Nd4j.create(new int[]{VECTOR_SIZE}, 'f'); // ordering is set as 'f', faster construct
//                label.putScalar(new int[] {0}, stock.getOpen());
//                label.putScalar(new int[] {1}, stock.getClose());
//                label.putScalar(new int[] {2}, stock.getLow());
//                label.putScalar(new int[] {3}, stock.getHigh());
//                label.putScalar(new int[] {4}, stock.getVolume());
//            } else {
//                label = Nd4j.create(new int[] {1}, 'f');
//                switch (category) {
//                    case OPEN: label.putScalar(new int[] {0}, stock.getOpen()); break;
//                    case CLOSE: label.putScalar(new int[] {0}, stock.getClose()); break;
//                    case LOW: label.putScalar(new int[] {0}, stock.getLow()); break;
//                    case HIGH: label.putScalar(new int[] {0}, stock.getHigh()); break;
//                    case VOLUME: label.putScalar(new int[] {0}, stock.getVolume()); break;
//                    default: throw new NoSuchElementException();
//                }
//            }
                testData.add(new Pair<>(input, label));
                addNextDay = true;
//                StockData stockData = new StockData();
//                stockData.setDate("Next Day");
//                test1.add(stockData);
            }
            log.info(testData.get(i).getKey().toString());
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1) * (max.getDouble(1) - min.getDouble(1)) + min.getDouble(1);
            actuals[i] = testData.get(i).getValue().getDouble(0);//test1.get(i + exampleLength - 1).getClose(); //testData.get(i).getValue().getDouble(0); ////
        }

//        log.info("Print out Predictions and Actual Values...");
        log.info("Date,Predict,Actual");
        for (int i = 0; i < predicts.length; i++)
            log.info(predicts[i] + "," + actuals[i]);
//            log.info(test1.get(i + exampleLength - 1).getDate() + ":" + predicts[i] + "," + actuals[i]);
        log.info("Plot...");
        PlotUtil.plot(predicts, actuals, String.valueOf(category));
    }


    private static void predictPriceMultiple(MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double max, double min) {
        // TODO
    }

    /**
     * Predict all the features (open, close, low, high prices and volume) of a stock one-day ahead
     */
    private static void predictAllCategories(MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, INDArray max, INDArray min) {
        int araySize = testData.size();
        if (addNextDay == false)
            araySize = testData.size() + 1;
        INDArray[] predicts = new INDArray[araySize];
        INDArray[] actuals = new INDArray[araySize];
        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
            actuals[i] = testData.get(i).getValue();
        }
        log.info("Print out Predictions and Actual Values...");
        log.info("Predict\tActual");
        for (int i = 0; i < predicts.length; i++) log.info(predicts[i] + "\t" + actuals[i]);
        log.info("Plot...");
        for (int n = 0; n < 5; n++) {
            double[] pred = new double[predicts.length];
            double[] actu = new double[actuals.length];
            for (int i = 0; i < predicts.length; i++) {
                pred[i] = predicts[i].getDouble(n);
                actu[i] = actuals[i].getDouble(n);
            }
            String name;
            switch (n) {
                case 0:
                    name = "Stock OPEN Price";
                    break;
                case 1:
                    name = "Stock CLOSE Price";
                    break;
                case 2:
                    name = "Stock LOW Price";
                    break;
                case 3:
                    name = "Stock HIGH Price";
                    break;
                case 4:
                    name = "Stock VOLUME Amount";
                    break;
                default:
                    throw new NoSuchElementException();
            }
            PlotUtil.plot(pred, actu, name);
        }
    }

}
