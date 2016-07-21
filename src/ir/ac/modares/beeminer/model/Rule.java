package ir.ac.modares.beeminer.model;

/**
 *
 * @author mehdi talebi
 */
public class Rule {

    public int trial;
    public double prob;
    public double fitness;
    public int foodClass;
    public Gene[] params;
    public int Dimensions;

    public Rule(int foodClass, int length) {
        trial = 0;
        prob = 0.1;
        fitness = 0;
        this.foodClass = foodClass;
        params = new Gene[length];
        this.Dimensions = length;
        for (int i = 0; i < Dimensions; i++) {
            params[i] = new Gene();
        }
    }

    public Rule() {
        trial = 0;
        prob = 0.1;
        fitness = 0;
        foodClass = 0;
        Dimensions = 3;
        params = new Gene[Dimensions];
        for (int i = 0; i < Dimensions; i++) {
            params[i] = new Gene();
        }
    }

    public double[] getParamInDouble() {
        double[] res = new double[Dimensions * 3];
        for (int i = 0; i < Dimensions; i++) {
            res[3 * i] = params[i].o;
            res[3 * i + 1] = params[i].l;
            res[3 * i + 2] = params[i].u;
        }

        return res;
    }

    public void setParamsWithDouble(double[] doubleParams) {
        for (int i = 0; i < Dimensions; i++) {
            params[i].o = doubleParams[3 * i];
            params[i].l = doubleParams[3 * i + 1];
            params[i].u = doubleParams[3 * i + 2];
        }
    }

    @Override
    public Rule clone() {
        Rule tmp = new Rule(foodClass, Dimensions);
        tmp.fitness = this.fitness;
        Gene tmpParam[] = new Gene[Dimensions];
        for (int i = 0; i < Dimensions; i++) {
            tmpParam[i] = new Gene();
            tmpParam[i] = params[i].clone();
        }
        tmp.params = tmpParam;
        tmp.prob = prob;
        tmp.trial = trial;
        return tmp;
    }

    public Rule clone(double[] param) {
        Rule tmp = new Rule(foodClass, Dimensions);
        tmp.fitness = this.fitness;
        tmp.setParamsWithDouble(param);
        tmp.prob = prob;
        tmp.trial = trial;
        return clone();
    }

    public void init() {
        for (int i = 0; i < Dimensions; i++) {
            params[i].l = Math.random();
            params[i].u = Math.random();
            params[i].o = Math.random();
        }
    }

    public void init(Rule model) {
        init();
        double U = Math.random();
        for (int i = 0; i < Dimensions; i++) {
            params[i].l += (model.params[i].l - params[i].l) * U;
            params[i].o += (model.params[i].o - params[i].o) * U;
            params[i].u += (model.params[i].u - params[i].u) * U;
        }

//        for(int i = 0; i < Dimensions ;i++)
//        { 
//            params[i] = Math.random();
//            if(params[i]<model.params[i])
//            {
//                params[i] = Math.log(model.params[i]/(model.params[i] -params[i] )) ;                     
//            }
//            else if(params[i]==model.params[i]){}
//            else
//            {
//                params[i] = Math.log(( 1- model.params[i])/(params[i]- model.params[i])) ; 
//            }
//        }
    }

    public void init(double[] U, double L[]) {

        for (int i = 0; i < Dimensions; i++) {
            params[i].o = Math.random();
            params[i].l = L[i];
            params[i].u = U[i];
        }
    }

    public void init(double R[], double L[], double U[]) {
        for (int i = 0; i < Dimensions; i++) {
            params[i].o = Math.random();
            params[i].l = L[i];
            params[i].u = U[i];
        }
    }

}
