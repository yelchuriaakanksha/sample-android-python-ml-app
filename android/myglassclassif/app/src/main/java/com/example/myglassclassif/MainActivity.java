package com.example.myglassclassif;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.text.InputFilter;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONException;
import org.json.JSONObject;

public class MainActivity extends AppCompatActivity
{
    EditText refid, sod, mag, alu, sil, pot, cal, bar, iro;
    Button sub, Can;
    String rindex;
    String sodium, magnesium, aluminium, silicon, pottasium, calcium, barium, iron;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        refid = findViewById(R.id.ri);
        sod = findViewById(R.id.na);
        mag = findViewById(R.id.mg);
        alu = findViewById(R.id.al);
        sil = findViewById(R.id.si);
        pot = findViewById(R.id.k);
        cal = findViewById(R.id.ca);
        bar = findViewById(R.id.ba);
        iro = findViewById(R.id.fe);
        sub = findViewById(R.id.submit);
        Can = findViewById(R.id.cancel);

        Can.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                rindex = refid.getText().toString();
                sodium = sod.getText().toString();
                magnesium = mag.getText().toString();
                aluminium = alu.getText().toString();
                silicon = sil.getText().toString();
                pottasium = pot.getText().toString();
                calcium = cal.getText().toString();
                barium = bar.getText().toString();
                iron = iro.getText().toString();
                if (rindex.isEmpty() && sodium.isEmpty() && magnesium.isEmpty() && aluminium.isEmpty() && silicon.isEmpty() && pottasium.isEmpty() && calcium.isEmpty() && barium.isEmpty() && iron.isEmpty()) {
                    Toast.makeText(getApplicationContext(), "Already empty", Toast.LENGTH_SHORT).show();
                } else {
                    refid.setText("");
                    sod.setText("");
                    mag.setText("");
                    alu.setText("");
                    sil.setText("");
                    pot.setText("");
                    cal.setText("");
                    bar.setText("");
                    iro.setText("");
                }
            }
        });

        refid.setFilters(new InputFilter[]{new MinMaxFilter(this, 1.5, 1.6)});
        sod.setFilters(new InputFilter[]{new MinMaxFilter(this, 12, 16)});
        mag.setFilters(new InputFilter[]{new MinMaxFilter(this, 0, 5)});
        alu.setFilters(new InputFilter[]{new MinMaxFilter(this, 0, 4)});
        sil.setFilters(new InputFilter[]{new MinMaxFilter(this, 69, 75)});
        pot.setFilters(new InputFilter[]{new MinMaxFilter(this, 0, 7)});
        cal.setFilters(new InputFilter[]{new MinMaxFilter(this, 5, 15)});
        bar.setFilters(new InputFilter[]{new MinMaxFilter(this, 0, 4)});
        iro.setFilters(new InputFilter[]{new MinMaxFilter(this, 0, 2)});


        sub.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                rindex = refid.getText().toString();
                sodium = sod.getText().toString();
                magnesium = mag.getText().toString();
                aluminium = alu.getText().toString();
                silicon = sil.getText().toString();
                pottasium = pot.getText().toString();
                calcium = cal.getText().toString();
                barium = bar.getText().toString();
                iron = iro.getText().toString();

                if (!(TextUtils.isEmpty(rindex)) && !(TextUtils.isEmpty(sodium)) && !(TextUtils.isEmpty(magnesium)) && !(TextUtils.isEmpty(aluminium)) && !(TextUtils.isEmpty(silicon)) && !(TextUtils.isEmpty(pottasium)) && !(TextUtils.isEmpty(calcium)) && !(TextUtils.isEmpty(barium)) && !(TextUtils.isEmpty(iron))) {
                    // Toast.makeText(this, "All details are ok", Toast.LENGTH_SHORT).show();
                    RequestQueue requestQueue = Volley.newRequestQueue(this);
                    final String url = "";
                    JSONObject postParams = new JSONObject();
                    try {
                        postParams.put("Refractive Index", rindex);
                        postParams.put("Sodium", sodium);
                        postParams.put("'Magnesium", magnesium);
                        postParams.put("Aluminium", aluminium);
                        postParams.put("Silicon", silicon);
                        postParams.put("Potassium", pottasium);
                        postParams.put("Calcium", calcium);
                        postParams.put("Barium", barium);
                        postParams.put("Iron", iron);
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }

                    JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(Request.Method.POST, url, postParams, new Response.Listener<JSONObject>() {
                        @Override
                        public void onResponse(JSONObject response) {
                            Log.i("On Response", "onResponse: " + response.toString());

                        }
                    }, new Response.ErrorListener() {
                        @Override
                        public void onErrorResponse(VolleyError error) {
                            Log.i("On Error", error.toString());
                            Toast.makeText(MainActivity.this, "" + error.toString(), Toast.LENGTH_SHORT).show();

                        }
                    });
                    requestQueue.add(jsonObjectRequest);
                } else {
                    Toast.makeText(this, "make sure all the details are given correctly", Toast.LENGTH_SHORT).show();
                }
            }
        });


    }


}

        //Toast.makeText(getApplicationContext(),"Cancel button Clicked",Toast.LENGTH_LONG).show();
        //Log.i("Cancelled ","Cancel button Clicked!")
