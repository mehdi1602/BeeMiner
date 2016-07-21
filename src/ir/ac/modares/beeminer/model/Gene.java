package ir.ac.modares.beeminer.model;

/**
 *
 * @author mehdi talebi
 */
public class Gene {

    public double o = 0;
    public double l = 0;
    public double u = 0;

    public boolean checkBounds(int subParamIndex, double lb, double ub) {
        boolean res = false;
        switch (subParamIndex) {
            case 0: {
                if (o < lb) {
                    o = lb;
                    res = true;
                }
                if (o > ub) {
                    o = ub;
                    res = true;
                }
                break;
            }
            case 1: {
                if (l < lb) {
                    this.l = lb;
                    res = true;
                }
                if (l > ub) {
                    this.l = ub;
                    res = true;
                }
                break;
            }
            case 2: {
                if (u < lb) {
                    this.u = lb;
                    res = true;
                }
                if (u > ub) {
                    this.u = ub;
                    res = true;
                }
                break;
            }
        }
        return res;
    }

    public double SubParam(int subParamIndex) {
        switch (subParamIndex) {
            case 0:
                return o;
            case 1:
                return l;
            case 2:
                return u;
            default:
                return o;
        }
    }

    public boolean change(int subParamIndex, double change, double vBest, boolean useVBest) {
        switch (subParamIndex) {
            case 0: {
                if (o >= 0.25 || !useVBest) {
                    o += change;
                } else {
                    o += change;
                    if (o >= 0.25 && o < 0.50) {
                        u = vBest;
                    } else if (o >= 0.50 && o < 0.75) {
                        l = vBest;
                    } else {
                        if (Math.random() > 0.5) {
                            l = vBest;
                            l = 0.0;
                        } else {
                            l = 1.0;
                            l = vBest;
                        }
                    }
                    System.out.println("vBest is: " + vBest);
                }
                break;
            }
            case 1: {
                l += change;
                break;
            }
            case 2: {
                u += change;
                break;
            }
            default:
                return false;
        }
        return true;
    }

    public boolean setParam(int subParamIndex, double change, double vBest, boolean useVBest) {
        switch (subParamIndex) {
            case 0: {
                if (o >= 0.25 || !useVBest) {
                    o = change;
                } else {
                    o = change;
                    if (o >= 0.25 && o < 0.50) {
                        u = vBest;
                    } else if (o >= 0.50 && o < 0.75) {
                        l = vBest;
                    } else {
                        if (Math.random() > 0.5) {
                            l = vBest;
                            l = 0.0;
                        } else {
                            l = 1.0;
                            l = vBest;
                        }
                    }
                }
                break;
            }
            case 1: {
                l = change;
                break;
            }
            case 2: {
                u = change;
                break;
            }
            default:
                return false;
        }
        return true;
    }

    @Override
    public Gene clone() {
        Gene res = new Gene();
        res.l = l;
        res.u = u;
        res.o = o;
        return res;
    }

    public Gene clone(double addValue) {
        Gene res = this.clone();

        return res;
    }
}
