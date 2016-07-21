package ir.ac.modares.beeminer.controller;

/**
 * 
 *
 * @author mehdi talebi
 */

import ir.ac.modares.beeminer.model.Rule;
import ir.ac.modares.beeminer.model.Attribute;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import weka.core.Instances;
import weka.core.converters.AbstractFileLoader;

public final class BeeMiner {

    // <editor-fold defaultstate="collapsed" desc="variacles">
    public boolean print = true;
    public boolean seperatTestFile = false;
    public boolean selectAtrTournoment = false;
    public boolean allNeighborForCategoricalAtr = true;
    public boolean setNeighborValueForCategorical = false;
    public boolean oneAgainstAll = false;
    public boolean useVBest = false;
    public boolean mongoliaAttack = false;
    public boolean localSearchOnNewFoods = false;
    public boolean localSearchOnBestFood = false;
    public boolean localSearchOnRandomFoods = true;
    public boolean incrementalPopulation = false;
    public boolean socialLearning = false;
    public double randomProbability = 0.03;
    public boolean sendScoutBasedOnGlobalMin = false;
    public boolean powellLS = true;
    public boolean simplexLS = false;
    public boolean normalLS = false;
    public int maxFitnessEvaluation = 100000;
    public int fitnessEvaluation = 0;
    public int changedOptimaEvaluationNo = 0;
    public double TestOnTrain = 0;
    public int incrementalPopulationLimit = maxFitnessEvaluation / 100;
    public int NP = 20; /* The number of colony size (employed bees+onlooker bees)*/

    public int FoodNumber; /*The number of food sources equals the half of the colony size*/

    public int limit = 400;  /*A food source which could not be improved through "limit" trials is abandoned by its employed bee*/

    public int foldNo = 0;
    public boolean variableClass = false;
    public double probForChangeCLass = 0.0;
    public int maxCycle; /*The number of cycles for foraging {a stopping criteria}*/

    public int reinitializedfoods = 0;
    /* Problem specific variables*/
    public int D; /*The number of parameters of the problem to be optimized*/

    public double lb; /*lower bound of the parameters. */

    public double ub; /*upper bound of the parameters. lb and ub can be defined as arrays for the problems of which parameters have different bounds*/

    public int RuleForClassifyCounter = 0;
    public int TermsForRulesCounter = 0;
    public int runtime;  /*Algorithm can be run many times in order to see its robustness*/

    public Rule Foods[];       /*Foods is the population of food sources. Each row of Foods matrix is a vector holding D parameters to be optimized. The number of rows of Foods matrix equals to the FoodNumber*/

    public Instances Instances;
    public Rule solution;            /*New solution (neighbour) produced by v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) j is a randomly chosen parameter and k is a randomlu chosen solution different from i*/


    public double ObjValSol;              /*Objective function value of new solution*/

    public double FitnessSol;              /*Fitness value of new solution*/

    public int neighbour, param2change;                   /*param2change corrresponds to j, neighbour corresponds to k in equation v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})*/

    public double GlobalMin;   /*Optimum solution obtained by ABC algorithm*/

    public double classEnt; /* Entorpy of class*/

    public Attribute Attributes[];
    public double gainSum = 0;
    public Rule GlobalParams = new Rule();                   /*Parameters of the optimum solution*/

    public double GlobalMins[] = new double[runtime];
    /*GlobalMins holds the GlobalMin of each run in multiple runs*/
    public double r; /*a random number in the range [0,1)*/

    public int iter;
    public int rep[];

    public int[] testIndex;//index of instances for testig algorithm
    public boolean[] classifiedTest = new boolean[9];//index of classified Test Set
    public int dataSetSize;//the size of all traning set
    public double[][] dataSet = new double[dataSetSize][D + 1];//dataset was loaded to the memory
    public boolean[][] missing;// this matrix shows the validation of value of the dataset(missing value or not)
    public int classCounter = 2;//number of diffrent classes in the dataset
    public int classIndex = 60;// the index of class attribute in dataset
    public int[] classInstancesCounter = new int[classCounter];//number of records corresponding to each class
    public boolean[] prunedIndex = new boolean[dataSetSize];//index of pruned instances
    public List<Rule> inductedRules = new ArrayList<>(classCounter);
    public List<Rule> rulePool = new ArrayList<>();
    private int testPoint = 0;
    public int[][] confMatrix;
    public List<String> classNames;
    public int improveFactor = 10;
    public int MissionClass = 0;
    //</editor-fold>

    public String Situation() {
        StringBuilder sb = new StringBuilder();
        StringBuilder //<editor-fold defaultstate="collapsed" desc="comment">
                append
                //</editor-fold>
                = sb.append("\n selectAtrTournoment: ").append(selectAtrTournoment)
                .append("\n allNeighborForCategoricalAtr: ").append(allNeighborForCategoricalAtr)
                .append("\n setNeighborValueForCategorical: ").append(setNeighborValueForCategorical)
                .append("\n oneAgainstAll: ").append(oneAgainstAll)
                .append("\n useVBest: ").append(useVBest)
                .append("\n localSearchOnRandomFoods: ").append(localSearchOnRandomFoods)
                .append("\n incrementalPopulation: ").append(incrementalPopulation)
                .append("\n socialLearning: ").append(socialLearning).append("\n powellLS: ")
                .append(powellLS).append("\n normalLS: ").append(normalLS)
                .append("\n sendScoutBasedOnGlobalMin: ").append(sendScoutBasedOnGlobalMin);

        return sb.toString();
    }

    public double[][] findMaxMin(double array[][], int Class) {
        int width = array[0].length;
        int length = array.length;
        double MaxMinArray[][] = new double[width][2];
        for (int i = 0; i < width; i++) {
            if (Attributes[i].attributeTypeInt > 0) {
                Attributes[i].atrMin = MaxMinArray[i][0] = 0;
                Attributes[i].atrMax = MaxMinArray[i][1] = Instances.attribute(i).numValues();
                continue;
            }

            MaxMinArray[i][0] = Double.MAX_VALUE;
            MaxMinArray[i][1] = Double.MIN_VALUE;

            for (int j = 1; j < length; j++) {

                if (Double.isNaN(array[j][i])) {
                    continue;
                }
                if (MaxMinArray[i][0] > array[j][i] && (Class == array[j][classIndex] || Class == -1)) {
                    MaxMinArray[i][0] = array[j][i];
                }
                if (MaxMinArray[i][1] < array[j][i] && (Class == array[j][classIndex] || Class == -1)) {
                    MaxMinArray[i][1] = array[j][i];
                }
            }
            if (Class == -1) {
                Attributes[i].atrMin = MaxMinArray[i][0];
                Attributes[i].atrMax = MaxMinArray[i][1];
            }
        }

        return MaxMinArray;
    }

    public BeeMiner(AbstractFileLoader loader) {
        classNames = new ArrayList<>(4);
        LoadFromFile2(loader);
        findMaxMin(dataSet, -1);
        Normalize(dataSet);
        hashingDataSet();
        FoodNumber = NP / 2;
        maxCycle = 1000;
        testPoint = 0;
        lb = 0;
        ub = 1.0;
        runtime = 3;
        Foods = new Rule[FoodNumber];
        ObjValSol = 0;
        FitnessSol = 0;
        neighbour = -1;
        param2change = -1;
        GlobalMin = -1;
        GlobalMins = new double[runtime];
        r = 0;
        testIndex = new int[dataSetSize / 10];
        classifiedTest = new boolean[dataSetSize / 10 + 1];
        confMatrix = new int[classCounter][classCounter];
        prunedIndex = new boolean[dataSetSize];
        inductedRules = new ArrayList<>(classCounter);
        rep = new int[D + 1];

    }

    public void initialization() {
        prunedIndex = new boolean[dataSetSize];
        inductedRules.clear();
        setTestIndexes();
    }

    public void hashingDataSet() {
        int index;
        double swap;
        boolean boolswap;
        for (int i = 0; i < dataSetSize; i++) {
            index = (int) (Math.random() * dataSetSize);
            for (int j = 0; j < D + 1; j++) {
                swap = dataSet[i][j];
                boolswap = missing[i][j];
                dataSet[i][j] = dataSet[index][j];
                missing[i][j] = missing[index][j];
                dataSet[index][j] = swap;
                missing[index][j] = boolswap;
            }
        }
    }

    public int Class_With_Max_Instance() {
        int ret = 0;
        int[] clsCnt = new int[classCounter];
        int cls;
        for (int i = 0; i < dataSetSize; i++) {
            if (isInArray(testIndex, i) || prunedIndex[i]) {
                continue;
            }
            cls = (int) (dataSet[i][classIndex] - 1);
            clsCnt[cls]++;
        }
        int max = 0;
        for (int i = 0; i < classCounter; i++) {
            if (max < clsCnt[i]) {
                max = clsCnt[i];
                ret = i + 1;
            }
        }
        return ret;
    }

    public int Class_With_Min_Instance() {
        int ret = 0;
        int[] clsCnt = new int[classCounter];
        int cls;
        for (int i = 0; i < dataSetSize; i++) {
            if (isInArray(testIndex, i) || prunedIndex[i]) {
                continue;
            }
            cls = (int) (dataSet[i][classIndex] - 1);
            clsCnt[cls]++;
        }
        int min = dataSetSize;
        for (int i = 0; i < classCounter; i++) {
            if (min > clsCnt[i] & clsCnt[i] != 0) {
                min = clsCnt[i];
                ret = i + 1;
            }
        }
        return ret;
    }

    public void pruneAll() {
        for (int i = 0; i < Foods.length; i++) {
            prune(Foods[i]);
        }
    }

    public int[][] Run() {
        int run = 0;
        int j = 0;
        double mean = 0;
        System.out.println(Situation());
        while (testPoint < dataSetSize) {
            iter = 0;
            run = 0;
            j = 0;
            mean = 0;
            initialization();
            int prunedCounter = 0, itr = 0;
            while (prunedCounter < 0.98 * (dataSetSize - testIndex.length)) {
                MissionClass = itr % classCounter + 1;//Class_With_Min_Instance();//Class_With_Max_Instance();//
                itr++;
                initial();
//                MemorizeBestSource();

                for (iter = 0; fitnessEvaluation < maxFitnessEvaluation//  iter<maxCycle
                        ; iter++) {
                    pruneAll();
                    SendEmployedBees();
                    CalculateProbabilities();
                    SendOnlookerBees();
                    if (incrementalPopulation) {
                        IncrementPopulation();
                    }
                    MemorizeBestSource();
                    SendScoutBees();
                    if (mongoliaAttack) {
                        mongoliaAttack();
                    }
                }

                if (print) {
                    System.out.println("Iter = " + iter + " evaluation# = " + fitnessEvaluation);
                }
                fitnessEvaluation = 0;
                prunedCounter = prunedCounter + pruning();

            }
            addLastRule();
            RuleForClassifyCounter += inductedRules.size();
            for (int r = 0; r < inductedRules.size(); r++) {
                TermsForRulesCounter += ruleLength(inductedRules.get(r));

            }
            double mean1 = Test();
            double mean2 = TestOnTrainingData();
            this.TestOnTrain = mean2;
            if (print) {
                System.out.println("reinitilized Times: " + reinitializedfoods);
                reinitializedfoods = 0;
                System.out.println("Test On Train Data Results: " + mean2);
                System.out.println((run + 1) + "***********************. accuracy = " + mean1);
            }
            GlobalMins[run] = mean1;
            mean = mean + mean1;
//                for(int y = 0; y < D; y++)
//                    System.out.println("rep: " + rep[y]);
        }
        return confMatrix;

    }

    public void mongoliaAttack() {
        if (fitnessEvaluation > 0.9 * maxFitnessEvaluation) {
            Foods[0] = GlobalParams.clone();
            for (int i = 1; i < Foods.length; i++) {
                init(i, true);
            }
            mongoliaAttack = false;
        }

    }

    public void IncrementPopulation() {
        if (fitnessEvaluation - changedOptimaEvaluationNo >= incrementalPopulationLimit) {
            FoodNumber++;
            Rule tmpFoods[] = new Rule[FoodNumber];
            for (int i = 0; i < FoodNumber - 1; i++) {
                tmpFoods[i] = Foods[i];
            }
            Foods = tmpFoods;
            changedOptimaEvaluationNo = fitnessEvaluation;
            Foods[FoodNumber - 1] = new Rule(GlobalParams.foodClass, D);
            init(FoodNumber - 1, true);
        }

    }

    public boolean LoadFromFile2(AbstractFileLoader loader) {
        String filePath = loader.retrieveFile().getAbsolutePath();
        try {
            System.out.println("file: " + loader.retrieveFile().getAbsolutePath());
            Instances = loader.getDataSet();
        } catch (IOException ex) {
            ex.printStackTrace();
            return false;
        }
        if (print) {
            System.out.println("relationship: " + Instances.relationName() + " file name: " + loader.retrieveFile().getName());
        }
        D = Instances.numAttributes() - 1;
        dataSetSize = Instances.numInstances();
        //0 = numeric -- 1 = categorical 
        Attributes = new Attribute[D + 1];
        Instances.setClassIndex(Instances.numAttributes() - 1);

        for (int i = 0; i < Instances.numAttributes(); i++) {
            Attributes[i] = new Attribute();
            Attributes[i].attributeTypeInt = Instances.attribute(i).type();
            if (Attributes[i].attributeTypeInt > 0) {
                Enumeration<String> val = Instances.attribute(i).enumerateValues();
                Attributes[i].categoricalValues = new ArrayList<>();
                for (; val.hasMoreElements();) {
                    Attributes[i].categoricalValues.add(val.nextElement());
                }
            }
            Attributes[i].atrMax = Instances.attribute(i).getUpperNumericBound();
            Attributes[i].atrMin = Instances.attribute(i).getLowerNumericBound();
            Attributes[i].attributeName = Instances.attribute(i).name();
            if (Attributes[i].attributeName.equalsIgnoreCase("class")) {
                Instances.setClassIndex(i);
            }

        }
        classIndex = Instances.classIndex();
        classNames = Attributes[classIndex].categoricalValues;
        classCounter = classNames.size();
        if (print) {
            System.out.println("class counter: " + classCounter);
        }
        dataSet = new double[dataSetSize][D + 1];
        missing = new boolean[dataSetSize][D + 1];
        boolean Missing;
        for (int i = 0; i < dataSetSize; i++) {
            for (int j = 0; j < D + 1; j++) {
                Missing = false;
                if (Instances.instance(i).isMissing(j)) {
                    Missing = true;
//                        System.out.println("value " + Instances.instance(i).value(j));
                }
                if (Attributes[j].attributeTypeInt == 0) {
                    dataSet[i][j] = Instances.instance(i).value(j);
                } else {
                    dataSet[i][j] = Instances.instance(i).value(j) + 1;
                }
                missing[i][j] = Missing;
//                    if(!Missing && Double.isNaN(dataSet[i][j]))
//                    {
//                        System.out.println(" nagoooooooooooo00000000000000000000000000000000000000000");
//                    }
            }

        }
        classIntstancesCounter();
        return true;
    }

    public void setClassIndex(int clasIdx) {
        Instances.setClassIndex(clasIdx);
        classIndex = clasIdx;
    }

    private void Normalize(double array[][]) {
        int length = array.length;
        int width = array[0].length;
        double[][] MinMaxArray = findMaxMin(array, -1);

        for (int i = 0; i < length; i++) {
            for (int j = 0; j < width; j++) {
                if (j != classIndex) {
                    array[i][j] = (array[i][j] - MinMaxArray[j][0]) / (MinMaxArray[j][1] - MinMaxArray[j][0]);
                }
            }
        }

    }

    private double reqInfo() {
        double res = 0;
        double p;
        double cof = Math.log10(2);
        int classRemainedInstance[] = new int[classCounter];
        int remainedInstances = 0;
        for (int n = 0; n < dataSetSize; n++) {
            if (prunedIndex[n] || isInArray(testIndex, n)) {
                continue;
            }
            classRemainedInstance[(int) dataSet[n][classIndex] - 1]++;// finding the class of pointing instance in case of any error there is some distance between "@data" tag and data
            remainedInstances++;
        }

        for (int i = 0; i < classCounter; i++) {
            if (classRemainedInstance[i] == 0) {
                continue;
            }
            p = (double) classRemainedInstance[i] / (double) remainedInstances;
            res += -1 * p * (Math.log10(p) / cof);
        }
        classEnt = res;
        return res;
    }

    private double binaryReqInfo(int classIdx) {
        double res = 0;
        double p;
        double cof = Math.log10(2);
        int classRemainedInstance[] = new int[2];
        int remainedInstances = 0;
        for (int n = 0; n < dataSetSize; n++) {
            if (prunedIndex[n] || isInArray(testIndex, n)) {
                continue;
            }
            if (dataSet[n][classIndex] == MissionClass) {
                classRemainedInstance[0]++;
            } else {
                classRemainedInstance[1]++;
            }
            remainedInstances++;
        }
        for (int i = 0; i < 2; i++) {
            if (classRemainedInstance[i] == 0) {
                continue;
            }
            p = (double) classRemainedInstance[i] / (double) remainedInstances;
            res += -1 * p * (Math.log10(p) / cof);
        }

        classEnt = res;
        return res;
    }

    private double Info(int[] p1) {
        if (p1[0] == 0) {
            return 0;//Empty Set
        }
        double p[] = new double[p1.length - 1];
        for (int i = 1; i < p1.length; i++) {
            p[i - 1] = (double) p1[i] / (double) p1[0];
        }

        double info = 0;
        double cof = Math.log10(2);
        for (int i = 0; i < p.length; i++) {
            if (p[i] == 0) {
                continue;
            }
            info += p[i] * (Math.log10(p[i]) / cof);// sum[p[i] * log2(p[i])]
        }
//        System.out.println("info: " + p1[2]);
        return -1 * info;
    }

    private double gainRatio(int classIdx, int atrIdx) {
        reqInfo();
        int shift = 0;
        if (atrIdx > classIdx) {
            shift++;
        }
        double res = 0, splitInfo = 0;
        int difValCnt;
        int[][] atrCnt;
        int atrValue, classValue;
        int nonePrunedData = 0;
        double minEnt = 100;
//    if(false)//Attributes[atrIdx].attributeType.equalsIgnoreCase("real")||  Attributes[atrIdx].attributeType.equalsIgnoreCase("numeric")||Attributes[atrIdx].attributeType.equalsIgnoreCase("integer"))
//    {
//        difValCnt = 4;
//        atrCnt = new int[difValCnt][classCounter+1];
//        for(int i =0; i < dataSetSize ; i++)
//        {
//            if(!prunedIndex[i] && !isInArray(testIndex, i))
//            {
//                atrValue = (int)Math.floor(dataSet[i][atrIdx]*difValCnt);
//                if(atrValue==difValCnt)atrValue--;
//                classValue = (int)dataSet[i][classIdx];
//                atrCnt[atrValue][0]++;
//                atrCnt[atrValue][classValue]++;
//                nonePrunedData++;
//            }
//        }
//        double p;
////                                            |Ti|
////	Info(X,T) = Sum for i from 1 to n of  ---- * Info(Ti)
////					      |T|
//        for(int i = 0; i < difValCnt; i++)
//        {
//            p = (double)atrCnt[i][0]/(double)nonePrunedData;
//            if(p != 0)
//            splitInfo -= p*Math.log(p)/Math.log(2);
//            res += p*Info(atrCnt[i]);
//        }
//        if(classEnt == 0) classEnt =  res*2;
//        Attributes[atrIdx-shift].atrGainRatio=classEnt-res;
//        if(splitInfo == 0) splitInfo=1.0f/difValCnt;
//        if(Attributes[atrIdx-shift].atrGainRatio == 0)Attributes[atrIdx-shift].atrGainRatio = 1.0f/difValCnt;
//        Attributes[atrIdx-shift].atrGainRatio=Attributes[atrIdx-shift].atrGainRatio/splitInfo;  
//              
//    }
//    else 
        if (Attributes[atrIdx].attributeTypeInt == 0) {

            for (int v = 0; v < dataSetSize; v++) {
                if (!prunedIndex[v] && !isInArray(testIndex, v)) {
                    continue;
                }
                difValCnt = 2;
                nonePrunedData = 0;
                res = 0;
                atrCnt = new int[difValCnt][classCounter + 1];
                for (int i = 0; i < dataSetSize; i++) {
                    if (!prunedIndex[i] && !isInArray(testIndex, i) && !missing[i][atrIdx]) {
                        if (dataSet[i][atrIdx] >= dataSet[v][atrIdx]) {
                            atrValue = 1;
                        } else {
                            atrValue = 0;
                        }
                        classValue = (int) dataSet[i][classIdx];
                        atrCnt[atrValue][0]++;
                        atrCnt[atrValue][classValue]++;
                        nonePrunedData++;
                    }
                }
                double p;
                //                                            |Ti|
                //	Info(X,T) = Sum for i from 1 to n of  ---- * Info(Ti)
                //					      |T|
                for (int i = 0; i < difValCnt; i++) {
                    p = (double) atrCnt[i][0] / (double) nonePrunedData;
                    res += p * Info(atrCnt[i]);
                }
                if (res < minEnt) {
                    minEnt = res;
                    Attributes[atrIdx - shift].vBest = dataSet[v][atrIdx];
                }
            }

            if (classEnt == 0) {
                Attributes[atrIdx - shift].atrGainRatio = 0.5;
            } else {
                Attributes[atrIdx - shift].atrGainRatio = classEnt - res;
            }

        } else if (Attributes[atrIdx].attributeTypeInt > 0) {
            difValCnt = Attributes[atrIdx].categoricalValues.size();
            atrCnt = new int[difValCnt][classCounter + 1];
            for (int i = 0; i < dataSetSize; i++) {
                if (!prunedIndex[i] && !isInArray(testIndex, i) && !missing[i][atrIdx]) {
                    atrValue = (int) Math.ceil(dataSet[i][atrIdx] * difValCnt) - 1;
                    classValue = (int) dataSet[i][classIdx];
                    atrCnt[atrValue][0]++;
                    atrCnt[atrValue][classValue]++;
                    nonePrunedData++;
                }

            }
            double p;
            for (int i = 0; i < difValCnt; i++) {
                p = (double) atrCnt[i][0] / (double) nonePrunedData;
                res += p * Info(atrCnt[i]);
            }
            if (classEnt == 0) {
                Attributes[atrIdx - shift].atrGainRatio = 0.5;
            } else {
                Attributes[atrIdx - shift].atrGainRatio = classEnt - res;
            }
        } else {
            System.out.println("pa chichi shodee?=====================================================================" + " type: " + Attributes[atrIdx].attributeType);
            return 0;
        }

        Attributes[atrIdx - shift].atrEnt = 0;//classBasedEntropy(MissionClass, atrIdx);
        Attributes[atrIdx - shift].atrBenefit = Attributes[atrIdx - shift].atrGainRatio;
        gainSum += Attributes[atrIdx - shift].atrBenefit;
        Attributes[atrIdx - shift].atrGainBound = gainSum;
//        if(print)
//            System.out.println("gain bound:  " + atrIdx + " = " + Attributes[atrIdx-shift].atrGainRatio +" vBest: " + Attributes[atrIdx-shift].vBest);
        return res;
    }

    private double binaryEntropoy(int classIdx, int atrIdx) {
        double ent = 0;
        classEnt = binaryReqInfo(MissionClass);
        int shift = 0;
        if (atrIdx > classIdx) {
            shift++;
        }
        double res = 0, splitInfo = 0;
        int difValCnt;
        int[][] atrCnt;
        int atrValue, classValue;
        int nonePrunedData = 0;
        double minEnt = 2;
//    if(false)//Attributes[atrIdx].attributeType.equalsIgnoreCase("real")||  Attributes[atrIdx].attributeType.equalsIgnoreCase("numeric")||Attributes[atrIdx].attributeType.equalsIgnoreCase("integer"))
//    {
//        difValCnt = 4;
//        atrCnt = new int[difValCnt][classCounter+1];
//        for(int i =0; i < dataSetSize ; i++)
//        {
//            if(!prunedIndex[i] && !isInArray(testIndex, i))
//            {
//                atrValue = (int)Math.floor(dataSet[i][atrIdx]*difValCnt);
//                if(atrValue==difValCnt)atrValue--;
//                classValue = (int)dataSet[i][classIdx];
//                atrCnt[atrValue][0]++;
//                atrCnt[atrValue][classValue]++;
//                nonePrunedData++;
//            }
//        }
//        double p;
////                                            |Ti|
////	Info(X,T) = Sum for i from 1 to n of  ---- * Info(Ti)
////					      |T|
//        for(int i = 0; i < difValCnt; i++)
//        {
//            p = (double)atrCnt[i][0]/(double)nonePrunedData;
//            if(p != 0)
//            splitInfo -= p*Math.log(p)/Math.log(2);
//            res += p*Info(atrCnt[i]);
//        }
//        if(classEnt == 0) classEnt =  res*2;
//        Attributes[atrIdx-shift].atrGainRatio=classEnt-res;
//        if(splitInfo == 0) splitInfo=1.0f/difValCnt;
//        if(Attributes[atrIdx-shift].atrGainRatio == 0)Attributes[atrIdx-shift].atrGainRatio = 1.0f/difValCnt;
//        Attributes[atrIdx-shift].atrGainRatio=Attributes[atrIdx-shift].atrGainRatio/splitInfo;  
//              
//    }
//    else 
        if (Attributes[atrIdx].attributeTypeInt == 0) {

            for (int v = 0; v < dataSetSize; v++) {
                if (!prunedIndex[v] && !isInArray(testIndex, v)) {
                    continue;
                }
                difValCnt = 2;
                nonePrunedData = 0;
                res = 0;
                atrCnt = new int[difValCnt][3];
                for (int i = 0; i < dataSetSize; i++) {
                    if (!prunedIndex[i] && !isInArray(testIndex, i)) {
                        if (dataSet[i][atrIdx] >= dataSet[v][atrIdx]) {
                            atrValue = 1;
                        } else {
                            atrValue = 0;
                        }
                        classValue = (int) dataSet[i][classIdx];
                        atrCnt[atrValue][0]++;
                        if (classValue == MissionClass) {
                            atrCnt[atrValue][1]++;
                        } else {
                            atrCnt[atrValue][2]++;
                        }
                        nonePrunedData++;

                    }
                }
                double p;
                //                                            |Ti|
                //	Info(X,T) = Sum for i from 1 to n of  ---- * Info(Ti)
                //					      |T|
                for (int i = 0; i < difValCnt; i++) {
                    p = (double) atrCnt[i][0] / (double) nonePrunedData;
                    res += p * Info(atrCnt[i]);
                }
                if (res < minEnt) {
                    minEnt = res;
                    Attributes[atrIdx - shift].vBest = dataSet[v][atrIdx];
                }
            }

            if (classEnt == 0) {
                ent = 0.5;
            } else {
                ent = classEnt - res;
            }

        } else if (Attributes[atrIdx].attributeTypeInt > 0) {
            difValCnt = Attributes[atrIdx].categoricalValues.size();
            atrCnt = new int[difValCnt][3];
            for (int i = 0; i < dataSetSize; i++) {

                if (prunedIndex[i] || isInArray(testIndex, i) || missing[i][atrIdx]) {
                    continue;
                }

                atrValue = (int) Math.ceil(dataSet[i][atrIdx] * difValCnt) - 1;
                classValue = (int) dataSet[i][classIdx];
                atrCnt[atrValue][0]++;
                if (classValue == MissionClass) {
                    atrCnt[atrValue][1]++;
                } else {
                    atrCnt[atrValue][2]++;
                }
                nonePrunedData++;

//            System.out.println("atrcnt : " + atrCnt[atrValue][2]);
            }
            double p;
            for (int i = 0; i < difValCnt; i++) {
                p = (double) atrCnt[i][0] / (double) nonePrunedData;
                res += p * Info(atrCnt[i]);
            }
            if (classEnt == 0) {
                ent = 0.5;
            } else {
                ent = classEnt - res;
            }
        } else {
            System.out.println("pa chichi shodee?=====================================================================" + " type: " + Attributes[atrIdx].attributeType);
            return 0;
        }
        ent += 0.1;
        Attributes[atrIdx - shift].atrEnt = 0;//classBasedEntropy(MissionClass, atrIdx);
        Attributes[atrIdx - shift].atrBenefit = Attributes[atrIdx - shift].atrGainRatio = ent;
        gainSum += Attributes[atrIdx - shift].atrBenefit;
        Attributes[atrIdx - shift].atrGainBound = gainSum;
        if (ent < 0) {
            if (print) {
                System.out.println("binary gained info:  " + atrIdx + " = " + classEnt + " vBest: " + Attributes[atrIdx - shift].vBest + " res " + res);
            }
        }

        return ent;
    }

    private double classBasedEntropy(int Class, int atrIdx) {
        int shift = 0;
        if (atrIdx > classIndex) {
            shift++;
        }
        if (Attributes[atrIdx].attributeTypeInt == 0) {
            double res = 0, splitInfo = 0;
            int difValCnt = 4;
            int[] atrCnt = new int[difValCnt];
            int atrValue, classValue;
            int nonePrunedData = 0;
            for (int i = 0; i < dataSetSize; i++) {
                if (!prunedIndex[i] && !isInArray(testIndex, i) && dataSet[i][classIndex] == Class) {
                    atrValue = (int) Math.floor(dataSet[i][atrIdx] * difValCnt);
                    if (atrValue == difValCnt) {
                        atrValue--;
                    }
                    atrCnt[atrValue]++;
                    nonePrunedData++;
                }
            }
            double p;
            if (nonePrunedData == 0) {
                return 0;
            }

            for (int i = 0; i < difValCnt; i++) {
                p = (double) atrCnt[i] / (double) nonePrunedData;
                if (p != 0) {
                    res -= p * Math.log(p) / Math.log(2);
                }
            }
            res = Math.log(difValCnt) / Math.log(2) - res;
            return res;
        }

        if (Attributes[atrIdx].attributeTypeInt > 0) {
            double res = 0;
            int difValCnt = Attributes[atrIdx].categoricalValues.size();
            int[] atrCnt = new int[difValCnt];
            int atrValue, classValue;
            int nonePrunedData = 0;
            for (int i = 0; i < dataSetSize; i++) {
                if (!prunedIndex[i] && !isInArray(testIndex, i) && dataSet[i][classIndex] == Class) {
                    atrValue = (int) Math.floor(dataSet[i][atrIdx] * (difValCnt - 1));
                    atrCnt[atrValue]++;
                    nonePrunedData++;
                }
            }
            double p;
            if (nonePrunedData == 0) {
                return 0;
            }

            for (int i = 0; i < difValCnt; i++) {
                p = (double) atrCnt[i] / (double) nonePrunedData;
                if (p != 0) {
                    res -= p * Math.log(p) / Math.log(2);
                }
            }
            res /= Math.log(difValCnt) / Math.log(2);
            return res;
        } else {
            System.out.println("pa chichi shodee?=====================================================================");
            return 0;
        }

    }

    private int selectAtr(double r) {
        if (selectAtrTournoment) {
            int r1 = (int) (Math.random() * D);
            int r2 = (int) (Math.random() * D);
            if (Attributes[r1].atrGainRatio > Attributes[r2].atrGainRatio) {
                return r1;
            } else {
                return r2;
            }
        }

        if (r < 0.00) {
            return -1;
        }
        r *= gainSum;
        int atrIdx;
        for (atrIdx = 0; atrIdx < D; atrIdx++) {
            if (Attributes[atrIdx].atrGainBound >= r) {
                break;
            }
        }
        rep[atrIdx]++;
        if (atrIdx == D) {
            atrIdx--;
        }
        return atrIdx;
    }

    public void classIntstancesCounter() {
        if (print) {
            System.out.println("classIndex: " + classIndex);
        }
        classInstancesCounter = new int[classCounter];
        for (int i = 0; i < classCounter; i++) {
            for (int j = 0; j < dataSetSize; j++) {
                if (dataSet[j][classIndex] == i + 1) {
                    classInstancesCounter[i]++;
                }
            }
        }

        for (int i = 0; i < classCounter; i++) {
            if (print) {
                System.out.println("class: " + classNames.get(i) + " = " + classInstancesCounter[i]);
            }
        }
    }

    public void setTestIndexes() {
        int notCompleteNo = 10 - (dataSetSize % 10);
        int[] testIndexes;
        if (notCompleteNo == 10) {
            notCompleteNo = 0;
            testIndexes = new int[dataSetSize / 10];
        } else {
            if (foldNo < 10 - notCompleteNo) {
                testIndexes = new int[dataSetSize / 10 + 1];
            } else {
                testIndexes = new int[dataSetSize / 10];
            }
        }

        int testInd = 0;
        for (int te = testPoint; te < testPoint + testIndexes.length; te++) {
            testIndexes[testInd] = te;
            testInd++;
        }

        testIndex = testIndexes;
        testPoint = testPoint + testIndexes.length;
        classifiedTest = new boolean[testIndexes.length];
        foldNo++;
    }

    /*The best food source is memorized*/
    void MemorizeBestSource() {

        int alterIndex = -1;

        for (int i = 0; i < FoodNumber; i++) {
            if (localSearchOnRandomFoods && Math.random() < randomProbability) {
                if (simplexLS) {
                    simplexLocalSearch(Foods[i], i);
                }
                if (powellLS) {
                    powellLocalSearch(Foods[i], i);
                }
                if (normalLS) {
                    normalLocalSearch(Foods[i], 0.5);
                }
            }
            if (Foods[i].fitness > GlobalMin) {
                GlobalMin = Foods[i].fitness;
                alterIndex = i;
            }

        }

        if (alterIndex >= 0) {
//               if(localSearchOnBestFood)               
//                   {
//                        if(simplexLS) simplexLocalSearch(Foods[alterIndex],alterIndex);
//                        if(powellLS) powellLocalSearch(Foods[alterIndex],alterIndex);
//                  }

            GlobalMin = Foods[alterIndex].fitness;
            GlobalParams = Foods[alterIndex].clone();

            if (print) {
                System.out.println("globalmin changed: " + GlobalMin + " Eval# = " + fitnessEvaluation);
            }
            changedOptimaEvaluationNo = fitnessEvaluation;

        }

    }

    void init(int index, boolean incrementedPop) {
        reinitializedfoods++;
        if (Foods[index] == null) {
            Foods[index] = new Rule(MissionClass, D);
        }

        int shift = 0;
        if (incrementedPop) {
            if (socialLearning) {
                Foods[index].init(GlobalParams);
            } else {
                if (index == Foods.length - 1) {
                    Foods[index].init();
                } else {
                    Foods[index].init(GlobalParams);
                }
            }

            for (int i = 0; i < Foods[index].Dimensions; i++) {
                Foods[index].params[i].checkBounds(0, lb, ub);
                Foods[index].params[i].checkBounds(1, lb, ub);
                Foods[index].params[i].checkBounds(2, lb, ub);
            }

        } else if (false) {

            double[] L = new double[D];
            double[] U = new double[D];
            double[] R = new double[D];
            shift = 0;
            for (int i = 0; i < D; i++) {
                if (i == classIndex) {
                    shift++;
                }
                R[i] = Math.random();
                if (index == 1) {
                    L[i] = Attributes[i + shift].vBest;
                } else {
                    L[i] = Math.random();
                }
                U[i] = Math.random();

            }
            Foods[index].init(R, L, U);
        } else {
            double[] L = new double[D];
            double[] U = new double[D];
            double[] R = new double[D];
            shift = 0;
            for (int i = 0; i < D; i++) {
                if (i == classIndex) {
                    shift = 1;
                }
                R[i] = Math.random();
                L[i] = Attributes[i + shift].vBest;
                U[i] = Attributes[i + shift].vBest;
            }
            Foods[index].init(R, L, U);
        }
        prune(Foods[index]);
        Foods[index].fitness = solutionQuality(Foods[index]);
        if (localSearchOnNewFoods) {
            if (simplexLS) {
                simplexLocalSearch(Foods[index], index);
            }
            if (powellLS) {
                powellLocalSearch(Foods[index], index);
            }
            if (normalLS) {
                normalLocalSearch(Foods[index], 0.5);
            }
        }

        Foods[index].trial = 0;
    }
    /*All food sources are initialized */

    void initial() {
        reqInfo();
        gainSum = 0;
        for (int y = 0; y < D; y++) {

            if (y < classIndex) {
                if (!oneAgainstAll) {
                    gainRatio(classIndex, y);
                } else {
                    binaryEntropoy(classIndex, y);
                }
            } else {
                if (!oneAgainstAll) {
                    gainRatio(classIndex, y + 1);
                } else {
                    binaryEntropoy(classIndex, y + 1);
                }
            }
        }
        fitnessEvaluation = 0;
        changedOptimaEvaluationNo = 0;
        FoodNumber = NP / 2;
        Foods = new Rule[FoodNumber];
        int i;
        for (i = 0; i < FoodNumber; i++) {
            init(i, false);
        }

//            MemorizeBestSource();
        GlobalMin = Foods[0].fitness;
        GlobalParams = Foods[0].clone();

    }

    private double changeAmount(int firstFoodIndex, int secondFoodIndex, int param2change, int subParam) {// to change this function for diffrentiation between numerical and categorical attributes
        r = Math.random();
        double change = 0;
        if (Attributes[param2change].attributeTypeInt > 0 && allNeighborForCategoricalAtr) {
            return (Math.random() - 0.5) * 2;
        } else if (Attributes[param2change].attributeTypeInt > 0 && setNeighborValueForCategorical) {
            return Foods[secondFoodIndex].params[param2change].SubParam(subParam);
        }
//       if(param2change%3 != 0)
//         change = r;
//       else
//       {
//        int shift = 0;
//        if(param2change>=classIndex)shift++;           
//        if(Attributes[param2change%D/3+shift].attributeType.equalsIgnoreCase("real")||
//               Attributes[param2change%D/3+shift].attributeType.equalsIgnoreCase("numeric")||Attributes[param2change%D/3+shift].attributeType.equalsIgnoreCase("integer"))
        switch (subParam) {
            case 0:
                change = (Foods[firstFoodIndex].params[param2change].o - Foods[secondFoodIndex].params[param2change].o) * (r - 0.5) * 2;
                break;
            case 1:
                change = (Foods[firstFoodIndex].params[param2change].l - Foods[secondFoodIndex].params[param2change].l) * (r - 0.5) * 2;
                break;
            case 2:
                change = (Foods[firstFoodIndex].params[param2change].u - Foods[secondFoodIndex].params[param2change].u) * (r - 0.5) * 2;
                break;
        }

//        else if(Attributes[param2change%D/3+shift].attributeType.equalsIgnoreCase("categorical"))
//            {
//                change = r;
//            }
//       }
        return change;
    }

    void SendEmployedBees() {
        int i, j, subParam2Change;
        double change, orgParam;
        /*Employed Bee Phase*/
        for (i = 0; i < FoodNumber; i++) {
            /*The parameter to be changed is determined randomly*/
            r = Math.random();
            param2change = selectAtr(r);
            subParam2Change = (int) (Math.random() * 3);

            if (param2change == -1) {
                int newClass = (int) (Math.random() * classCounter) + 1;
                orgParam = Foods[i].foodClass;
                Foods[i].foodClass = newClass;

            } else {
                orgParam = Foods[i].params[param2change].SubParam(subParam2Change);
                /*A randomly chosen solution is used in producing a mutant solution of the solution i*/
                r = Math.random();
                neighbour = (int) (r * FoodNumber);

                /*Randomly selected solution must be different from the solution i*/
                while (neighbour == i) {
                    r = Math.random();
                    neighbour = (int) (r * FoodNumber);
                }

                change = changeAmount(i, neighbour, param2change, subParam2Change);

                Foods[i].params[param2change].change(subParam2Change, change, Attributes[param2change].vBest, useVBest);
                /*if generated parameter value is out of boundaries, it is shifted onto the boundaries*/

                Foods[i].params[param2change].checkBounds(subParam2Change, lb, ub);
            }

            FitnessSol = solutionQuality(Foods[i]);

            /*a greedy selection is applied between the current solution i and its mutant*/
            if (FitnessSol > Foods[i].fitness) {
                Foods[i].trial = 0;
                Foods[i].fitness = FitnessSol;
            } else {   /*if the solution i can not be improved, increase its trial counter*/

                if (param2change == -1) {
                    Foods[i].foodClass = (int) orgParam;
                }
                Foods[i].trial++;
                if (FitnessSol < Foods[i].fitness && param2change != -1) {
                    Foods[i].params[param2change].setParam(subParam2Change, orgParam, Attributes[param2change].vBest, useVBest);
                }
            }
        }

        /*end of employed bee phase*/
    }

    void CalculateProbabilities() {
        int i;
        double maxfit;
        maxfit = Foods[0].fitness;
        for (i = 1; i < FoodNumber; i++) {
            if (Foods[i].fitness > maxfit) {
                maxfit = Foods[i].fitness;
            }
        }

        for (i = 0; i < FoodNumber; i++) {
            if (maxfit != 0) {
                Foods[i].prob = (0.9 * (Foods[i].fitness / maxfit)) + 0.1;
            } else {
                Foods[i].prob = 0.1;
            }
        }
    }

    public int selectFoodSource() {
        double r = Math.random();
        double r1 = Math.random();
        int food1 = (int) (r * FoodNumber);
        int food2 = (int) (r * FoodNumber);
        if (Foods[food1].fitness > Foods[food2].fitness) {
            return food1;
        } else {
            return food2;
        }
    }

    void SendOnlookerBees() {

        int i, j, t, subParam2Change;
        i = 0;
        t = 0;
        /*onlooker Bee Phase*/
        while (t < FoodNumber) {
//        i = selectFoodSource();
            r = Math.random();

            if (r < Foods[i].prob) /*choose a food source depending on its probability to be chosen*/ {
                t++;

                /*The parameter to be changed is determined randomly*/
                r = Math.random();
                param2change = selectAtr(r);
                subParam2Change = (int) (Math.random() * 3);

                /*A randomly chosen solution is used in producing a mutant solution of the solution i*/
                r = Math.random();
                neighbour = (int) (r * FoodNumber);

                /*Randomly selected solution must be different from the solution i*/
                while (neighbour == i) {
                    r = Math.random();
                    neighbour = (int) (r * FoodNumber);
                }
                solution = Foods[i].clone();
                if (param2change == -1) {
                    int newClass = (int) (Math.random() * classCounter) + 1;
                    while (solution.foodClass == newClass) {
                        newClass = (int) (Math.random() * classCounter) + 1;
                    }
                    solution.foodClass = newClass;
                } else {
                    solution.params[param2change].change(subParam2Change, changeAmount(i, neighbour, param2change, subParam2Change), Attributes[param2change].vBest, useVBest);
                    solution.params[param2change].checkBounds(subParam2Change, lb, ub);
                }
                FitnessSol = solutionQuality(solution);

                /*a greedy selection is applied between the current solution i and its mutant*/
                if (FitnessSol > Foods[i].fitness) {
                    /*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
                    Foods[i].trial = 0;
                    Foods[i] = solution.clone();
                    Foods[i].fitness = FitnessSol;
                } //            else if(FitnessSol==Foods[i].fitness)
                //            {
                //                Foods[i].trial++;
                //                Foods[i] = solution.clone();
                //            }
                else {   /*if the solution i can not be improved, increase its trial counter*/

                    Foods[i].trial++;
                }
            } /*if */

            i = (i + 1) % (FoodNumber);
        }/*while*/

        /*end of onlooker bee phase     */
    }

    /*determine the food sources whose trial counter exceeds the "limit" value. In Basic ABC, only one scout is allowed to occur in each cycle*/
    void SendScoutBees() {
        int maxtrialindex, i;
        maxtrialindex = 0;
        for (i = 1; i < FoodNumber; i++) {
            if (Foods[i].trial > Foods[maxtrialindex].trial) {
                maxtrialindex = i;
            }
        }
        if (Foods[maxtrialindex].trial >= limit)//&& Math.random() > iter/maxCycle)
        {
            init(maxtrialindex, sendScoutBasedOnGlobalMin);
        }
    }

    public void YDM(List<Double> vec, List<double[]> rules, List<Integer> classes) {
        for (int i = 0; i < vec.size(); i++) {

            int idx = 0;
            int exClass = (i % classCounter) + 1;
            double max = -1;
            for (int k = i; k < vec.size(); k++) {
                if (classes.get(k) == exClass) {
                    if (vec.get(k) > max) {
                        idx = k;
                        max = vec.get(k);
                    }
                }
            }

            if (idx != i) {
                double tmp = vec.get(i);
                double[] rtmp = rules.get(i);
                int ctmp = classes.get(i);
                vec.remove(i);
                rules.remove(i);
                classes.remove(i);
                vec.add(i, vec.get(idx - 1));
                rules.add(i, rules.get(idx - 1));
                classes.add(i, classes.get(idx - 1));
                vec.remove(idx);
                rules.remove(idx);
                classes.remove(idx);
                vec.add(idx, tmp);
                rules.add(idx, rtmp);
                classes.add(idx, ctmp);
            }

            if (print) {
                System.out.println("class:" + i + " " + classNames.get(classes.get(i) - 1));
            }
            if (print) {
                System.out.println("accu:" + i + " " + vec.get(i));
            }
        }
    }

    public void Sort(List<Double> vec, List<double[]> rules, List<Integer> classes) {
        for (int i = 0; i < vec.size(); i++) {
            for (int j = i + 1; j < vec.size(); j++) {
                if (vec.get(j) > vec.get(i)) {
                    double tmp = vec.get(i);
                    double[] rtmp = rules.get(i);
                    int ctmp = classes.get(i);
                    vec.remove(i);
                    rules.remove(i);
                    classes.remove(i);
                    vec.add(i, vec.get(j - 1));
                    rules.add(i, rules.get(j - 1));
                    classes.add(i, classes.get(j - 1));
                    vec.remove(j);
                    rules.remove(j);
                    classes.remove(j);
                    vec.add(j, tmp);
                    rules.add(j, rtmp);
                    classes.add(j, ctmp);
                }
            }
            if (print) {
                System.out.println("class:" + i + " " + classNames.get(classes.get(i) - 1));
            }
        }
    }

    public double Test() {
        double res = 0;
        int corrCounter = 0;
        //   this.YDM(inductedRuleAccuracy, inductedRules, inductedRulesClass);

        int testCounter = 0;

        for (int r = 0; r < inductedRules.size(); r++) {

            for (int i = 0; i < testIndex.length; i++) {
                if (classifiedTest[i]) {
                    continue;
                }
                if (isInRuleDomain(inductedRules.get(r), dataSet[testIndex[i]], missing[testIndex[i]])) {
                    testCounter++;
                    int PC = inductedRules.get(r).foodClass;
                    int RC = (int) dataSet[testIndex[i]][classIndex];
                    if (PC == RC) {
                        corrCounter++;
                    }
                    confMatrix[RC - 1][PC - 1]++;
                    classifiedTest[i] = true;
                }
            }

        }
        res = (double) corrCounter;
        res = res / (double) testCounter;
        return res;
    }

    public double TestOnTrainingData() {
        double res = 0;
        int corrCounter = 0;
        int testCounter = 0;
        boolean classifiedTRA[] = new boolean[dataSetSize];

        for (int r = 0; r < inductedRules.size(); r++) {
            for (int i = 0; i < dataSetSize; i++) {
                if (isInArray(testIndex, i) || classifiedTRA[i]) {
                    continue;
                }
                if (isInRuleDomain(inductedRules.get(r), dataSet[i], missing[i])) {
                    testCounter++;
                    int PC = inductedRules.get(r).foodClass;
                    int RC = (int) dataSet[i][classIndex];
                    if (PC == RC) {
                        corrCounter++;
                    }
                    classifiedTRA[i] = true;
                }
            }
        }
        System.out.println("corr: " + corrCounter + " total: " + testCounter);
        res = (double) corrCounter;
        res = res / (double) testCounter;
        return res;
    }

    boolean isInRuleDomain(Rule rule, double[] Ins, boolean[] missing) {
        boolean A = true;
        int[] atrStatus = new int[D];
        int shift = 0, elementCeil, minCeil;
        for (int i = 0; i < D; i++) {

            if (rule.params[i].o < 0.25) {
                atrStatus[i] = 0;
            } else if (rule.params[i].o < 0.50) {
                atrStatus[i] = 1;
            } else if (rule.params[i].o < 0.75) {
                atrStatus[i] = 2;
            } else {
                atrStatus[i] = 3;
            }
        }

        for (int j = 0; j < D && A; j++) {
            if (j == classIndex) {
                shift++;
            }
            if (atrStatus[j] == 0) {
                continue;
            }

            if (Attributes[j + shift].attributeTypeInt == 0) {
                switch (atrStatus[j]) {
                    case 1:
                        if (Ins[j + shift] > rule.params[j].u || missing[j + shift]) {
                            A = false;
                        }
                        break;
                    case 2:
                        if (Ins[j + shift] < rule.params[j].l || missing[j + shift]) {
                            A = false;
                        }
                        break;
                    case 3:
                        if (Ins[j + shift] > rule.params[j].u || Ins[j + shift] < rule.params[j].l || missing[j + shift]) {
                            A = false;
                        }
                        break;
                }
            } else if (Attributes[j + shift].attributeTypeInt > 0) {
                elementCeil = (int) Math.ceil(Ins[j + shift] * Instances.attribute(j + shift).numValues());

                if (rule.params[j].l == 1) {
                    minCeil = Instances.attribute(j + shift).numValues();
                } else {
                    minCeil = (int) Math.floor(rule.params[j].l * Attributes[j + shift].categoricalValues.size()) + 1;
                }
                switch (atrStatus[j]) {
                    case 1:
                        if (elementCeil != minCeil || missing[j + shift]) {
                            A = false;
                        }
                        break;
                    case 0:
                        if (elementCeil == minCeil || missing[j + shift]) {
                            A = false;
                        }
                        break;
                }
            }
        }
        return A;
    }

    boolean isInRuleDomain(double[] Ins, boolean[] missing) {
        return isInRuleDomain(GlobalParams, Ins, missing);
    }

    double solutionQuality(Rule sol) {
        fitnessEvaluation++;
        int confsionMatrix[][];
        int ruleLength = ruleLength(sol);
        double maxAccuracy = -1, accu = 0;

        // for(int i = 0; i < classCounter; i++)
        {
            confsionMatrix = ConfiusionMatrix(sol, sol.foodClass);
            accu = ruleQuality(confsionMatrix, ruleLength);

            if (maxAccuracy < accu) {
                maxAccuracy = accu;

            }
        }

        return maxAccuracy;
    }

    int ruleLength(Rule rule) {
        int ruleLength = 0;
        for (int i = 0; i < D; i++) {
            double g = rule.params[i].o;
            if (g >= 0.25) {
                ruleLength++;
            }
        }
        return ruleLength;
    }

    int[][] ConfiusionMatrix(Rule sol, int predClass) {

        int res[][] = new int[2][2];
        //[TP TN]
        //[FP FN]
        boolean A, C;

        for (int i = 0; i < dataSetSize; i++) {
            if (isInArray(testIndex, i) || prunedIndex[i]) {
                continue;
            }
            A = isInRuleDomain(sol, dataSet[i], missing[i]);
            C = (dataSet[i][classIndex] == sol.foodClass);
            if (A && C) {
                res[0][0]++;
            }
            if (A && !C) {
                res[1][0]++;
            }
            if (!A && C) {
                res[1][1]++;
            }
            if (!A && !C) {
                res[0][1]++;
            }
            if (predClass == -1 && A) {
                prunedIndex[i] = true;
            }
        }
        return res;
    }

    boolean isInArray(int[] arra, int x) {
        boolean res = false;
        for (int i = 0; i < arra.length; i++) {
            if (arra[i] == x) {
                res = true;
            }
        }

        return res;
    }

    /**
     * this function try to decode the rules to be understood by human
     *
     * @param sol the rule that you want to have it's string
     * @return decoded input rule
     */
    String ruleString(Rule sol) {

        String res = ruleLength(sol) + " IF ";
        int[] atrStatus = new int[D];

        for (int i = 0; i < D; i++) {
            if (sol.params[i].o < 0.5) {
                atrStatus[i] = 0;
            } else if (sol.params[i].o < 0.75) {
                atrStatus[i] = 1;
            } else {
                atrStatus[i] = 2;
            }
        }

        int shift = 0, minCeil;
        for (int j = 0; j < D; j++) {

            if (j >= classIndex) {
                shift = 1;
            }
            if (sol.params[j].o < 0.25) {
                continue;
            }

            if (Attributes[j + shift].attributeTypeInt == 0) {
                switch (atrStatus[j]) {
                    case 0:
                        res = res + ", " + Attributes[j + shift].attributeName + " <= " + (sol.params[j].l * (Attributes[j + shift].atrMax - Attributes[j + shift].atrMin) + Attributes[j + shift].atrMin);
                        break;
                    case 1:
                        res = res + ", " + Attributes[j + shift].attributeName + " => " + (sol.params[j].l * (Attributes[j + shift].atrMax - Attributes[j + shift].atrMin) + Attributes[j + shift].atrMin);
                        break;
                    case 2:
                        res = res + ", " + Attributes[j + shift].attributeName + " BETWEEN " + (sol.params[j].l * (Attributes[j + shift].atrMax - Attributes[j + shift].atrMin) + Attributes[j + shift].atrMin) + " AND "
                                + (sol.params[j].u * (Attributes[j + shift].atrMax - Attributes[j + shift].atrMin) + Attributes[j + shift].atrMin);
                        break;
                }
            } else if (Attributes[j + shift].attributeTypeInt > 0) {
                if (sol.params[j].l == 1.0) {
                    minCeil = Instances.attribute(j + shift).numValues();
                } else {
                    minCeil = (int) Math.floor(sol.params[j].l * Instances.attribute(j + shift).numValues()) + 1;
                }
                switch (atrStatus[j]) {
                    case 0:
                        res = res + ", " + Attributes[j + shift].attributeName + " = " + Attributes[j + shift].categoricalValues.get(minCeil - 1);
                        break;
                    case 1:
                        res = res + ", " + Attributes[j + shift].attributeName + " != " + Attributes[j + shift].categoricalValues.get(minCeil - 1);
                        break;

                }
            }
        }

        res = res + " THEN Class " + classNames.get(sol.foodClass - 1);
        return res;
    }

    /**
     * this function will add last rule to the rule set. last rule is a rule
     * that cover all remained instances and it's class is the class with
     * maximum remained instance
     */
    public void addLastRule() {
        int max = 0;
        int lastClass = 1;
        for (int j = 0; j < classCounter; j++) {
            int counter = 0;
            for (int i = 0; i < dataSetSize; i++) {

                if (isInArray(testIndex, i) || prunedIndex[i]) {
                    continue;
                }

                if (dataSet[i][classIndex] == j + 1) {
                    counter++;
                }
            }
            if (max < counter) {
                lastClass = j + 1;
                max = counter;
            }
        }

        Rule lastRule = new Rule(lastClass, D);
        inductedRules.add(lastRule);
        if (print) {
            System.out.println("last rule: " + ruleString(lastRule) + " max remained data" + max);
        }
    }

    /**
     * this function will calculate the fitness of a rule with input features
     *
     * @param confsionMatrix the confusion matrix for one rule
     * @param ruleLength the length of that rule
     * @return the quality of input rule
     */
    public double ruleQuality(int confusionMatrix[][], int ruleLength) {

        int TP, TN, FP, FN;
        double accu = 0;
        TP = confusionMatrix[0][0];
        TN = confusionMatrix[0][1];
        FP = confusionMatrix[1][0];
        FN = confusionMatrix[1][1];
        double supp = 0, spec = 0, conf = 0;
        if ((TP + FN) != 0) {
            supp = ((double) TP) / (double) (TP + FN);
        }
        if ((TN + FP) != 0) {
            spec = ((double) TN) / (double) (TN + FP);
        }
        if ((TP + FP) != 0) {
            conf = (double) TP / (double) (TP + FP);
        }
        if ((TP + FP) != 0) {
            conf = (double) TP / (double) (TP + FP);
        }

        accu = spec * supp + 0.1 * 1 / (1 + ruleLength);

        return accu;
    }

    private double ruleCoveredSpace(Rule rule) {
        double coveredSpace = 1.0;
//         int attrStatus [] = new int[D/3];
//         for(int i = 0; i < D/3;i++)
//         {
//             attrStatus[i] = (int)Math.floor( rule.params[i]*4);
//             if(attrStatus[i] == 4) attrStatus[i] = 3;
//         }
//         for(int i = 0; i < D/3; i++)
//         {
//             if(attrStatus[i] == 0);
//             else if(attrStatus[i] == 1) coveredSpace *= rule.params[i+D/3];
//             else if(attrStatus[i] == 2) coveredSpace *= (1.0-rule.params[i+2*D/3]);
//             else if(attrStatus[i] == 3) coveredSpace *= (rule.params[i+2*D/3] -rule.params[i+D/3]);
//         }
//         coveredSpace *= Math.pow(2, ruleLength(rule));
        return coveredSpace;
    }

    private void generalizeRule(Rule rule) {
        int attrStatus[] = new int[D / 3];
        double orgValue, tmpValue, orgParam, tmpParam;
        System.out.println("before generalize: " + ruleCoveredSpace(rule));
//         for(int i = 0; i < D/3;i++)
//         {
//             attrStatus[i] = (int)Math.floor( rule.params[i]*4);
//             if(attrStatus[i] == 4) attrStatus[i] = 3;
//         }
//         for(int i = 0; i < D/3; i++)
//         {
//             if(attrStatus[i] == 0)continue;
//             if(attrStatus[i] ==1 ||attrStatus[i] ==3 )
//             {
//                 orgValue = solutionQuality(rule, -1);
//                 for(int x = 0; x<10;x++)
//                 {                    
//                     orgParam = rule.params[i+D/3];
//                     rule.params[i+D/3] += 0.05;
//                     if(rule.params[i+D/3] > 1) {rule.params[i+D/3] = 1; break;}
//                     tmpValue = solutionQuality(rule, -1);
//                     if(orgValue>tmpValue) {rule.params[i+D/3] = orgParam; break;}
//                     
//                 }
//             }
//             else if(attrStatus[i] ==2 ||attrStatus[i] ==3 )
//             {
//                orgValue = solutionQuality(rule, -1);
//                for(int x = 0; x<10;x++)
//                 {                     
//                     orgParam = rule.params[i+2*D/3];
//                     rule.params[i+2*D/3] -= 0.05;
//                     if(rule.params[i+2*D/3] < 0) {rule.params[i+2*D/3] = 0;break;}
//                     tmpValue = solutionQuality(rule, -1);
//                     if(orgValue>tmpValue) {rule.params[i+2*D/3] = orgParam; break;}
//                 }
//             }
//             
//         }
        System.out.println("after generalize: " + ruleCoveredSpace(rule));
    }

    public void prune(Rule rule) {
        double orgParam;
        double tmpAcuu, orgAcuu = solutionQuality(rule);
        for (int i = 0; i < D; i++) {
            orgParam = rule.params[i].o;
            for (int j = 0; j < 4; j++) {
                if ((rule.params[i].o - 0.25) < 0) {
                    break;
                }
                rule.params[i].o -= 0.25;
                tmpAcuu = solutionQuality(rule);
                if (tmpAcuu >= orgAcuu) {
                    orgAcuu = tmpAcuu;
                    orgParam = rule.params[i].o;
                }
            }
            rule.params[i].setParam(0, orgParam, 0, false);
            rule.fitness = orgAcuu;
        }

    }

    public int pruning() {
        prune(GlobalParams);
//             generalizeRule(GlobalParams);
        inductedRules.add(GlobalParams.clone());
        int[][] con = ConfiusionMatrix(GlobalParams, -1);
        int counter = con[0][0] + con[1][0];

        if (print) {
            System.out.println("\n\ncounter " + counter + " TP: " + con[0][0] + " FP: "
                    + con[1][0] + " testPoint: " + testPoint);
            System.out.println(ruleString(GlobalParams));
            System.out.println("Rule Accuracy: " + inductedRules.get(inductedRules.size() - 1).foodClass + " globalmin = " + GlobalMin + " FoodNumber: " + FoodNumber + "\n\n");

        }
//             atrGainBound = new double[D/3];
//             reqInfo();
//                      for(int y = 0; y<D/3; y++)
//         {
//             if(y < classIndex)
//                gainRatio(classIndex,y);
//             else
//                gainRatio(classIndex,y+1);
//         }
        return counter;

    }

    //<editor-fold defaultstate="collapsed" desc="LOCAL_SEARCH_PART"> 
    public void powellLocalSearch(Rule sol, int index) {
//         System.out.println("primary: " + Foods[index].fitness);
        double[][] Xi = new double[sol.Dimensions * 3][sol.Dimensions * 3];
        Rule tmpSol2;
        boolean isChanged[] = new boolean[sol.Dimensions * 3];
        Rule tmpSol;
        Rule primarySol = sol.clone();
        double bVal;
        for (int i = 0; i < sol.Dimensions * 3; i++) {
            Xi[i][i] = 1;
        }
        tmpSol = sol.clone();
        int j = 0;
        double firstValue, secondValue, valueDif = 0.0, primVal = Foods[index].fitness;
        int alternateIndex = 0;
        while (j < 3) {
            j++;
            for (int i = 0; i < tmpSol.Dimensions * 3; i++) {
                firstValue = Foods[index].fitness;
                if (isChanged[i]) {
                    tmpSol2 = lineMinimizationFibunacci(tmpSol, Xi[i], 0.5, index, 2);
                    if (tmpSol2 != null) {
                        tmpSol.setParamsWithDouble(tmpSol2.getParamInDouble());// System.arraycopy(tmpSol2.params, 0, tmpSol.params, 0, tmpSol.Dimensions);
                    }
                } else {
                    if (i % 3 == 0) {
                        tmpSol2 = lineMinimizationFibunacci(tmpSol, Xi[i], 0.5, index, 1);
                        if (tmpSol2 != null) {
                            System.arraycopy(tmpSol2.params, 0, tmpSol.params, 0, tmpSol.Dimensions);
                        }
                    } else if (tmpSol.params[i / 3].o < 0.25) {
                    } else if (i % 3 == 1 && tmpSol.params[i / 3].o >= 0.5 && tmpSol.params[i / 3].o < 0.75) {
                    } else if (i % 3 == 2 && tmpSol.params[i / 3].o < 0.5) {
                    } else {
                        tmpSol2 = lineMinimizationFibunacci(tmpSol, Xi[i], 0.5, index, 2);
                        if (tmpSol2 != null) {
                            tmpSol.setParamsWithDouble(tmpSol2.getParamInDouble());//System.arraycopy(tmpSol2.params, 0, tmpSol.params, 0, tmpSol.Dimensions);
                        }
                    }
                }

                secondValue = Foods[index].fitness;
                // valueDif = secondValue - firstValue;
                if (secondValue - firstValue > valueDif) {
                    alternateIndex = i;
                    valueDif = secondValue - firstValue;
                }
            }
            Xi[alternateIndex] = vecMinus(primarySol.getParamInDouble(), tmpSol.getParamInDouble());
            isChanged[alternateIndex] = true;
            tmpSol2 = lineMinimizationFibunacci(tmpSol, Xi[alternateIndex], 0.5, index, 2);
            if (tmpSol2 != null) {
                System.arraycopy(tmpSol2.params, 0, tmpSol.params, 0, tmpSol.Dimensions);
            }

            System.arraycopy(tmpSol.params, 0, primarySol.params, 0, tmpSol.Dimensions);

        }
        System.arraycopy(tmpSol.params, 0, sol.params, 0, tmpSol.Dimensions);
        bVal = Foods[index].fitness;
//          System.out.println("secondary: " + Foods[index].fitness);
//              if(primVal< Foods[index].fitness){
//               System.out.println("primary value: " + primVal + "secondary value: " + Foods[index].fitness);
//              }
    }

    /**
     *
     * @param p first point of the function
     * @param xit the direction of the search
     * @return best coefficient for local minimization with xit direction
     * @throws ArithmeticException
     */
    private Rule lineMinimizationFibunacci(Rule p, double[] xit, double length, int index, int minimizationMethod) {
        if (minimizationMethod == 1) {
            int atrIdx = 0;
            double primaryFitness, secondaryFitness, primaryValue;

            for (int i = 0; i < D; i++) {
                if (xit[i] != 0) {
                    atrIdx = i;
                }
            }

            int addValue = (int) Math.floor(p.params[atrIdx / 3].o * 4);
            if (addValue == 4) {
                addValue = 3;
            }
            for (int i = 0; i < 3; i++) {
                primaryFitness = Foods[index].fitness;
                addValue++;
                primaryValue = p.params[atrIdx / 3].o;
                p.params[atrIdx / 3].o = (addValue % 4) / 4.0f;
                secondaryFitness = solutionQuality(p);

                if (primaryFitness >= secondaryFitness) {
                    p.params[atrIdx / 3].o = primaryValue;
                } else {
                    Foods[index].fitness = secondaryFitness;
                }

            }
            return p;
        }

        double primaryValue = Foods[index].fitness;
        double a[], b[], c[], d[],
                b1[] = new double[p.Dimensions], a1[] = new double[p.Dimensions];
        Rule aRule, bRule, cRule = null, dRule = null;
        double rr, aVal, bVal, cVal = 0, dVal = 0;
        int fibunucci[] = new int[8];
        fibunucci[0] = 0;
        fibunucci[1] = 1;
        double delta = length;
        for (int i = 2; i < fibunucci.length; i++) {
            fibunucci[i] = fibunucci[i - 1] + fibunucci[i - 2];
        }
        a = vecSum(p.getParamInDouble(), multiplyVector(xit, -1 * length));
        b = vecSum(a, multiplyVector(xit, 2 * length));

        for (int i = 0; i < a.length; i++) // check boundaries
        {
            if (a[i] < lb) {
                a[i] = 0;
            }
            if (b[i] < lb) {
                b[i] = 0;
            }
            if (a[i] > ub) {
                a[i] = 1;
            }
            if (b[i] > ub) {
                b[i] = 1;
            }

        }
        aRule = p.clone(a);
        bRule = p.clone(b);
        aVal = solutionQuality(aRule);
        bVal = solutionQuality(bRule);
        System.arraycopy(a, 0, a1, 0, a1.length);
        System.arraycopy(b, 0, b1, 0, b1.length);

        for (int i = fibunucci.length - 1; i > 2; i--) {
            rr = (double) fibunucci[i - 1] / fibunucci[i];
            c = vecSum(a, multiplyVector(vecMinus(b, a), 1 - rr));
            d = vecSum(a, multiplyVector(vecMinus(b, a), rr));
            cRule = p.clone(c);
            dRule = p.clone(d);
            cVal = solutionQuality(cRule);
            dVal = solutionQuality(dRule);
            delta = delta * (1 - rr);

            if (cVal >= dVal) {
                System.arraycopy(d, 0, b, 0, b.length);

            } else {
                System.arraycopy(c, 0, a, 0, a.length);
            }

        }
//                if(print)
//                System.out.println("cval = " + cVal + " dval= " + dVal + " primary value is : " + primaryValue + " aval = " + 
//                        aVal + " bval = " + bVal + "delta = " + delta);
        if (aVal > bVal && aVal > cVal && aVal > dVal) {
            if (primaryValue > aVal) {
                return null;
            }
            Foods[index].fitness = aVal;
            aRule = p.clone(a1);
            return aRule;
        } else if (bVal > cVal && bVal > dVal) {
            if (primaryValue > bVal) {
                return null;
            }
            Foods[index].fitness = bVal;
            bRule = p.clone(b1);
            return bRule;
        } else if (cVal >= dVal) {
            if (primaryValue > cVal) {
                return null;
            }
            Foods[index].fitness = cVal;
            return cRule;
        } else {
            if (primaryValue > dVal) {
                return null;
            }
            Foods[index].fitness = dVal;
            return dRule;
        }
    }

    public double[] multiplyVector(double[] vec, double landa) {
        double res[] = new double[vec.length];
        for (int i = 0; i < vec.length; i++) {
            res[i] = vec[i] * landa;
        }
        return res;
    }

    public double[] vecSum(double[] a, double[] b) {
        double res[] = new double[a.length];
        if (a.length != b.length) {
            System.out.println("variable length vector for adding");
            return null;
        }
        for (int i = 0; i < b.length; i++) {
            res[i] = b[i] + a[i];
        }
        return res;

    }

    public double[] vecMinus(double[] a, double[] b) {
        double res[] = new double[a.length];
        if (a.length != b.length) {
            System.out.println("variable length vector for adding");
            return res;
        }
        for (int i = 0; i < b.length; i++) {
            res[i] = a[i] - b[i];
        }
        return res;

    }

    public void simplexLocalSearch(Rule sol, int index) {
        if (Foods[index].fitness == 1.0) {
            return;
        }

        int dimensionNo = sol.Dimensions;
        Rule neighbors[] = new Rule[dimensionNo + 1];
        for (int i = 0; i < sol.Dimensions; i++) {
            neighbors[i] = randomNeighbor(sol, 0.5, i);
        }
        neighbors[sol.Dimensions] = sol;

        double[] refledPoint, extendedPoint, meanPoint,
                MW, MR, neighborFitnesses = new double[dimensionNo + 1];
        for (int i = 0; i < dimensionNo; i++) {
            neighborFitnesses[i] = solutionQuality(neighbors[i]);
        }
        neighborFitnesses[dimensionNo] = Foods[index].fitness;

        double reflectedPointFitness, MWFitness, MRFitness, extendedPointFitness, primaryFitness = -1;
        int worstNeghborIndex = -1;
        int it = 0;
        while (it < 10) {
            it++;
            worstNeghborIndex = findMin(neighborFitnesses, primaryFitness, worstNeghborIndex);
            primaryFitness = neighborFitnesses[worstNeghborIndex];
            meanPoint = meanVector(neighbors, worstNeghborIndex);

            refledPoint = vecMinus(multiplyVector(meanPoint, 2), neighbors[worstNeghborIndex].getParamInDouble());
            Rule reflectedPointRule = sol.clone(refledPoint);
            reflectedPointFitness = solutionQuality(reflectedPointRule);
            if (reflectedPointFitness > neighborFitnesses[worstNeghborIndex]) {
                neighbors[worstNeghborIndex] = reflectedPointRule;
                neighborFitnesses[worstNeghborIndex] = reflectedPointFitness;
                extendedPoint = vecMinus(multiplyVector(refledPoint, 2), meanPoint);
                Rule extendedPointRule = sol.clone(extendedPoint);
                extendedPointFitness = solutionQuality(extendedPointRule);
                if (extendedPointFitness > reflectedPointFitness) {
                    neighbors[worstNeghborIndex] = extendedPointRule;
                    neighborFitnesses[worstNeghborIndex] = extendedPointFitness;
                }
            } else {
                extendedPoint = vecMinus(multiplyVector(refledPoint, 2), meanPoint);
                MW = midPoint(extendedPoint, meanPoint);
                MR = midPoint(meanPoint, neighbors[worstNeghborIndex].getParamInDouble());
                Rule MWRule = sol.clone(MW);
                Rule MRRule = sol.clone(MR);
                MWFitness = solutionQuality(MWRule);
                MRFitness = solutionQuality(MRRule);

                if (MRFitness > MWFitness) {
                    if (MRFitness > neighborFitnesses[worstNeghborIndex]) {
                        neighbors[worstNeghborIndex] = MRRule;
                        neighborFitnesses[worstNeghborIndex] = MRFitness;
                    }
                }
                if (MWFitness >= MRFitness) {
                    if (MWFitness > neighborFitnesses[worstNeghborIndex]) {
                        neighbors[worstNeghborIndex] = MWRule;
                        neighborFitnesses[worstNeghborIndex] = MWFitness;
                    }
                }
                if (primaryFitness == neighborFitnesses[worstNeghborIndex]) {
                    Shrink(neighbors, neighborFitnesses, findMax(neighborFitnesses));
                }
            }
        }
        int maxIndex = findMax(neighborFitnesses);
        sol = neighbors[maxIndex];
        Foods[index].fitness = neighborFitnesses[maxIndex];

    }

    public void Shrink(Rule points[], double[] fitness, int maxFitnessIndex) {
        for (int i = 0; i < points.length; i++) {
            points[i].setParamsWithDouble(midPoint(points[i].getParamInDouble(), points[maxFitnessIndex].getParamInDouble()));//params = midPoint(points[i].params, points[maxFitnessIndex].params);
            fitness[i] = solutionQuality(points[i]);
        }
    }

    public int findMax(double[] array) {
        int minIndex = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > array[minIndex]) {
                minIndex = i;
            }
        }

        return minIndex;
    }

    public double[] midPoint(double[] Vec1, double[] Vec2) {
        double mid[] = new double[Vec1.length];
        for (int i = 0; i < Vec1.length; i++) {
            mid[i] = (Vec1[i] + Vec2[i]) / 2;
        }
        return mid;
    }

    public double[] meanVector(Rule vectors[], int worstNeighbor) {
        double sumVector[] = new double[vectors[0].Dimensions * 3];
        double meanVec[] = new double[vectors[0].Dimensions * 3];
        for (int i = 0; i < vectors[0].Dimensions; i++) {
            sumVector[i] = 0;
            for (int j = 0; j < vectors.length; j++) {
                if (j != worstNeighbor) {
                    sumVector[i * 3] += vectors[j].params[i].o;
                    sumVector[i * 3 + 1] += vectors[j].params[i].l;
                    sumVector[i * 3 + 2] += vectors[j].params[i].u;
                }
            }
            meanVec[i] = sumVector[i] / (vectors.length - 1);
        }
        return meanVec;
    }

    public int findMin(double[] array, double lowBound, int forbidenIndex) {
        int minIndex = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] < array[minIndex]) {
                if (array[i] >= lowBound) {
                    if (i != forbidenIndex) {
                        minIndex = i;
                    }
                }
            }
        }

        return minIndex;
    }

    public Rule randomNeighbor(Rule point, double neighborRange, int atrIdx) {
        Rule neighborMat = point.clone();
        double random;
        for (int i = 0; i < neighborMat.Dimensions; i++) {
            if (i == atrIdx) {
                random = (Math.random() - 0.5) * neighborRange;
                if (i % 3 == 0) {
                    neighborMat.params[i / 3].o = point.params[i].o + random;
                    neighborMat.params[i / 3].checkBounds(0, lb, ub);
                } else if (i % 3 == 1) {
                    neighborMat.params[i / 3].l = point.params[i].l + random;
                    neighborMat.params[i / 3].checkBounds(1, lb, ub);
                } else if (i % 3 == 2) {
                    neighborMat.params[i / 3].u = point.params[i].u + random;
                    neighborMat.params[i / 3].checkBounds(2, lb, ub);
                }
            } else {
                neighborMat.params[i] = point.params[i].clone();
            }
        }
        return neighborMat;
    }

    public void normalLocalSearch(Rule rule, double SD) {
        Random rand = new Random();
        double primaryFitness, secondaryFitness = 0, firstVal = 0;

        for (int i = 0; i < rule.Dimensions * 3; i++) {
            firstVal = rule.params[i / 3].SubParam(i % 3);
            primaryFitness = rule.fitness;
            if (i % 3 == 0) {
                rule.params[i / 3].change(0, rand.nextGaussian() * SD, Attributes[i / 3].vBest, useVBest);
                rule.params[i / 3].checkBounds(0, lb, ub);
            } else if (i % 3 == 1) {
                rule.params[i / 3].change(1, rand.nextGaussian() * SD, Attributes[i / 3].vBest, useVBest);
                rule.params[i / 3].checkBounds(1, lb, ub);
            } else if (i % 3 == 2) {
                rule.params[i / 3].change(2, rand.nextGaussian() * SD, Attributes[i / 3].vBest, useVBest);
                rule.params[i / 3].checkBounds(2, lb, ub);
            }
            secondaryFitness = solutionQuality(rule);
            if (secondaryFitness > primaryFitness) {
                rule.fitness = secondaryFitness;
            } else {
                rule.params[i / 3].setParam(i % 3, firstVal, Attributes[i / 3].vBest, useVBest);
            }
        }
    }

    //</editor-fold>
    public static void main(String[] o) {
        Random rand = new Random();
        System.out.println("rand: " + rand.nextGaussian());
    }
}
