package com.example.myglassclassif;

import android.content.Context;
import android.text.InputFilter;
import android.text.Spanned;
import android.widget.Toast;

public class MinMaxFilter implements InputFilter
{

        private final double minval;
        private final double maxval;
        private final Context context;

        public MinMaxFilter(MainActivity mainActivity, int minval, int maxval)
        {
            this.minval=minval;
            this.maxval=maxval;
            this.context=mainActivity;
        }
        public MinMaxFilter(MainActivity mainActivity,double minval, double maxval)
        {
            this.minval=minval;
            this.maxval=maxval;
            this.context=mainActivity;
        }
        @Override
        public CharSequence filter(CharSequence source, int start, int end, Spanned dest, int dstart, int dend)
        {
            try {
                double input = Double.parseDouble(dest.toString() + source.toString());
                if (isInRange(minval, maxval, input)) {
                    return null;
                } else {
                    Toast.makeText(context, "Enter between the range" + minval + " and " + maxval, Toast.LENGTH_SHORT).show();
                }
            } catch (NumberFormatException nfe) {
            }
            return "";
        }
        private boolean isInRange(double a, double b, double c)
        {
            return b > a ? c >= a && c <= b : c >= b && c <= a;
        }

    }



