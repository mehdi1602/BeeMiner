
package ir.ac.modares.beeminer.model;

import java.util.List;

/**
 *
 * @author mehdi talebi
 */
public class Attribute {

    public String attributeName;
    public String attributeType;
    public int attributeTypeInt;
    public List<String> categoricalValues;
    public double atrMax;
    public double atrMin;
    public double atrGainRatio;
    public double atrGainBound;
    public double atrEnt;
    public double atrBenefit;
    public double vBest;//represent the value that caused maximum entropy

    public double Parser(String atr, boolean missing) {
        double res = 0;
        if (missing) {
            return Math.random();
        }

        if (attributeTypeInt == 0) {

            res = Double.parseDouble(atr);
        } else if (attributeTypeInt > 0) {
            boolean flag = false;
            for (int i = 0; !flag && i < categoricalValues.size(); i++) {
                if (atr.equals(categoricalValues.get(i))) {
                    flag = true;
                    res = i + 1;
                }
            }
        }
        return res;
    }

}
