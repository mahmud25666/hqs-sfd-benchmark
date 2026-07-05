// esp32_edge_timing.ino
//
// Measures REAL ESP32 hardware inference latency for the distilled
// edge_model.h surrogate, to the same rigor bar as the rest of the paper
// (fixed CPU frequency, radios off, mean +/- std over independent batches,
// on-device accuracy check run in the same build that's timed).
//
// Setup:
//   1. Keep this .ino and edge_model.h in the same folder, named
//      esp32_edge_timing/ (Arduino requires folder name == .ino name).
//   2. Board: Tools > Board > ESP32 Dev Module (or your specific board).
//   3. Upload, then open Serial Monitor at 115200 baud.
//
// What changed vs. the first version:
//   - 200 real (Kn, AR, Freq, cd) rows sampled from master_sfd.csv
//     (fixed seed=42), embedded as PROGMEM arrays, instead of 6 fixed points.
//   - CPU frequency is explicitly set and confirmed, not just read.
//   - WiFi and Bluetooth radios are disabled before timing to remove
//     interrupt jitter.
//   - 5 independent timed batches -> mean +/- std, matching the paper's
//     seed-averaging convention elsewhere.
//   - On-device MSE against the true cd values is computed and printed
//     BEFORE timing, in the same binary, so a correctness regression on
//     this hardware/toolchain can't silently pass alongside a timing number.

#include "edge_model.h"
#include <WiFi.h>
#include <esp_bt.h>
#include <esp_timer.h>

// ---- Real dataset sample (200 rows, seed=42) ----
static const int N_SAMPLES = 200;

static const float SAMPLE_KN[N_SAMPLES] PROGMEM = {
    0.053443f, 0.209734f, 0.886847f, 0.120855f, 0.451617f, 0.087194f, 1.535882f, 0.078180f, 0.086978f, 1.086311f, 0.679306f, 0.167928f, 0.188849f, 0.176195f, 2.914242f, 0.219675f, 0.441189f, 0.886847f, 0.459340f, 0.108631f, 1.105353f, 0.153588f, 0.912596f, 0.818857f, 0.638251f, 0.589320f, 0.110535f, 0.086978f, 0.199212f, 0.094389f, 0.065678f, 0.441189f, 2.134102f, 1.030571f, 0.158097f, 1.211551f, 0.389354f, 0.070395f, 2.379140f, 0.058039f, 0.075173f, 1.086311f, 0.679306f, 0.118358f, 0.071991f, 0.627520f, 0.078180f, 0.296532f, 0.417701f, 0.193131f, 0.562652f, 0.562652f, 0.118358f, 0.078180f, 0.051811f, 0.389354f, 1.000315f, 0.219675f, 0.627520f, 0.145135f, 0.330580f, 0.086978f, 0.143197f, 0.065678f, 1.509424f, 0.061303f, 0.088685f, 0.219675f, 0.062752f, 0.104452f, 0.074258f, 0.742582f, 0.262405f, 0.135912f, 0.051811f, 0.219675f, 1.105353f, 0.182237f, 1.232269f, 2.379140f, 0.094389f, 0.057251f, 0.216347f, 0.121155f, 1.232269f, 0.364611f, 0.296532f, 0.097814f, 1.509424f, 0.164457f, 0.871936f, 0.198972f, 0.276804f, 0.075173f, 0.193131f, 0.781802f, 0.193131f, 0.138993f, 0.412031f, 0.131153f, 0.451617f, 0.351845f, 0.679306f, 0.534426f, 1.712233f, 0.199212f, 0.742582f, 0.153588f, 0.091260f, 0.324218f, 0.062752f, 0.078180f, 0.384155f, 0.851802f, 0.871936f, 0.199212f, 0.351845f, 0.818857f, 0.296532f, 0.067931f, 0.100032f, 0.054101f, 0.417701f, 0.216347f, 1.712233f, 1.535882f, 0.201665f, 1.712233f, 0.112056f, 0.518111f, 0.441189f, 0.087194f, 1.712233f, 0.412031f, 0.113780f, 0.135912f, 0.086978f, 0.051811f, 0.572515f, 0.276804f, 0.176195f, 0.085180f, 0.506625f, 0.074258f, 0.741687f, 0.108631f, 1.683445f, 0.233914f, 0.450499f, 0.818857f, 0.340179f, 0.233914f, 0.625966f, 0.058932f, 1.183575f, 0.097814f, 0.081886f, 0.085180f, 0.138993f, 0.110535f, 0.209734f, 0.656782f, 0.103057f, 0.291424f, 0.625966f, 0.638251f, 0.364611f, 0.150942f, 0.404933f, 0.074258f, 2.914242f, 0.058039f, 0.488887f, 0.851802f, 0.589320f, 0.123227f, 0.118358f, 0.441189f, 0.158097f, 0.198972f, 0.340179f, 1.683445f, 0.305237f, 0.135912f, 1.712233f, 0.253218f, 0.062597f, 0.209734f, 0.067931f, 0.050663f, 0.472677f, 0.625966f, 0.233914f, 0.058039f, 0.143370f, 0.051811f, 0.291424f, 0.168345f, 0.074169f, 1.509424f
};

static const float SAMPLE_AR[N_SAMPLES] PROGMEM = {
    35.000000f, 20.000000f, 15.000000f, 35.000000f, 30.000000f, 10.000000f, 40.000000f, 35.000000f, 30.000000f, 15.000000f, 20.000000f, 35.000000f, 40.000000f, 50.000000f, 5.000000f, 40.000000f, 45.000000f, 20.000000f, 20.000000f, 10.000000f, 40.000000f, 50.000000f, 10.000000f, 50.000000f, 20.000000f, 15.000000f, 25.000000f, 10.000000f, 45.000000f, 45.000000f, 50.000000f, 40.000000f, 40.000000f, 15.000000f, 40.000000f, 30.000000f, 5.000000f, 20.000000f, 5.000000f, 5.000000f, 5.000000f, 50.000000f, 40.000000f, 45.000000f, 30.000000f, 30.000000f, 45.000000f, 35.000000f, 10.000000f, 25.000000f, 15.000000f, 5.000000f, 10.000000f, 10.000000f, 45.000000f, 45.000000f, 30.000000f, 50.000000f, 20.000000f, 5.000000f, 30.000000f, 25.000000f, 25.000000f, 40.000000f, 25.000000f, 10.000000f, 50.000000f, 5.000000f, 45.000000f, 25.000000f, 25.000000f, 15.000000f, 35.000000f, 10.000000f, 45.000000f, 30.000000f, 20.000000f, 50.000000f, 45.000000f, 30.000000f, 5.000000f, 45.000000f, 20.000000f, 40.000000f, 50.000000f, 5.000000f, 50.000000f, 15.000000f, 10.000000f, 10.000000f, 30.000000f, 20.000000f, 30.000000f, 30.000000f, 45.000000f, 50.000000f, 50.000000f, 40.000000f, 35.000000f, 15.000000f, 25.000000f, 45.000000f, 25.000000f, 40.000000f, 45.000000f, 5.000000f, 30.000000f, 5.000000f, 40.000000f, 25.000000f, 10.000000f, 50.000000f, 25.000000f, 10.000000f, 25.000000f, 35.000000f, 30.000000f, 10.000000f, 40.000000f, 30.000000f, 15.000000f, 40.000000f, 15.000000f, 20.000000f, 35.000000f, 45.000000f, 20.000000f, 20.000000f, 50.000000f, 10.000000f, 25.000000f, 25.000000f, 40.000000f, 35.000000f, 20.000000f, 25.000000f, 15.000000f, 10.000000f, 35.000000f, 50.000000f, 5.000000f, 20.000000f, 35.000000f, 40.000000f, 25.000000f, 40.000000f, 20.000000f, 10.000000f, 40.000000f, 15.000000f, 45.000000f, 20.000000f, 40.000000f, 35.000000f, 50.000000f, 45.000000f, 35.000000f, 25.000000f, 45.000000f, 40.000000f, 35.000000f, 15.000000f, 50.000000f, 5.000000f, 50.000000f, 30.000000f, 10.000000f, 20.000000f, 45.000000f, 25.000000f, 50.000000f, 40.000000f, 25.000000f, 45.000000f, 45.000000f, 40.000000f, 15.000000f, 10.000000f, 45.000000f, 30.000000f, 35.000000f, 10.000000f, 5.000000f, 10.000000f, 10.000000f, 25.000000f, 15.000000f, 20.000000f, 25.000000f, 35.000000f, 40.000000f, 45.000000f, 25.000000f, 15.000000f, 35.000000f, 30.000000f, 20.000000f, 45.000000f, 5.000000f, 50.000000f
};

static const float SAMPLE_FREQ[N_SAMPLES] PROGMEM = {
    317300.858013f, 161792.021660f, 628318.530718f, 628318.530718f, 317300.858013f, 6283.185307f, 628318.530718f, 628318.530718f, 628318.530718f, 6283.185307f, 317300.858013f, 472809.694365f, 472809.694365f, 161792.021660f, 6283.185307f, 317300.858013f, 6283.185307f, 472809.694365f, 317300.858013f, 6283.185307f, 161792.021660f, 161792.021660f, 628318.530718f, 317300.858013f, 6283.185307f, 6283.185307f, 472809.694365f, 472809.694365f, 161792.021660f, 472809.694365f, 161792.021660f, 472809.694365f, 317300.858013f, 317300.858013f, 161792.021660f, 628318.530718f, 472809.694365f, 6283.185307f, 472809.694365f, 317300.858013f, 6283.185307f, 161792.021660f, 628318.530718f, 161792.021660f, 161792.021660f, 628318.530718f, 628318.530718f, 628318.530718f, 161792.021660f, 472809.694365f, 6283.185307f, 628318.530718f, 161792.021660f, 317300.858013f, 6283.185307f, 317300.858013f, 472809.694365f, 628318.530718f, 628318.530718f, 161792.021660f, 6283.185307f, 317300.858013f, 161792.021660f, 472809.694365f, 628318.530718f, 6283.185307f, 161792.021660f, 317300.858013f, 161792.021660f, 628318.530718f, 317300.858013f, 317300.858013f, 161792.021660f, 317300.858013f, 317300.858013f, 6283.185307f, 472809.694365f, 161792.021660f, 6283.185307f, 161792.021660f, 317300.858013f, 472809.694365f, 161792.021660f, 472809.694365f, 161792.021660f, 6283.185307f, 161792.021660f, 317300.858013f, 628318.530718f, 161792.021660f, 628318.530718f, 161792.021660f, 472809.694365f, 628318.530718f, 628318.530718f, 472809.694365f, 472809.694365f, 317300.858013f, 6283.185307f, 161792.021660f, 161792.021660f, 472809.694365f, 628318.530718f, 628318.530718f, 161792.021660f, 628318.530718f, 317300.858013f, 628318.530718f, 161792.021660f, 161792.021660f, 6283.185307f, 628318.530718f, 472809.694365f, 6283.185307f, 472809.694365f, 317300.858013f, 628318.530718f, 161792.021660f, 317300.858013f, 161792.021660f, 628318.530718f, 472809.694365f, 161792.021660f, 628318.530718f, 317300.858013f, 472809.694365f, 161792.021660f, 472809.694365f, 628318.530718f, 317300.858013f, 472809.694365f, 317300.858013f, 472809.694365f, 161792.021660f, 161792.021660f, 317300.858013f, 6283.185307f, 317300.858013f, 317300.858013f, 161792.021660f, 628318.530718f, 628318.530718f, 6283.185307f, 6283.185307f, 6283.185307f, 161792.021660f, 628318.530718f, 161792.021660f, 317300.858013f, 6283.185307f, 6283.185307f, 472809.694365f, 628318.530718f, 317300.858013f, 6283.185307f, 628318.530718f, 161792.021660f, 317300.858013f, 317300.858013f, 317300.858013f, 472809.694365f, 472809.694365f, 628318.530718f, 472809.694365f, 472809.694365f, 317300.858013f, 161792.021660f, 6283.185307f, 317300.858013f, 161792.021660f, 161792.021660f, 161792.021660f, 161792.021660f, 6283.185307f, 317300.858013f, 317300.858013f, 6283.185307f, 317300.858013f, 317300.858013f, 472809.694365f, 161792.021660f, 628318.530718f, 472809.694365f, 161792.021660f, 161792.021660f, 161792.021660f, 472809.694365f, 472809.694365f, 472809.694365f, 161792.021660f, 472809.694365f, 472809.694365f, 161792.021660f, 317300.858013f, 472809.694365f, 161792.021660f, 161792.021660f, 161792.021660f, 161792.021660f, 6283.185307f
};

static const float SAMPLE_CD[N_SAMPLES] PROGMEM = {
    134.316045f, 57.187509f, 2.309069f, 9.109804f, 24.000407f, 256.066700f, 5.812476f, 198.577563f, 13.713671f, 120.083582f, 1.842736f, 10.480269f, 5.397459f, 236.651595f, 1.657864f, 32.505285f, 3240.951034f, 7.250380f, 20.953772f, 355.803558f, 59.712918f, 838.030545f, 0.129970f, 6.908831f, 763.609311f, 62.196337f, 51.780769f, 1.260609f, 121.257017f, 35.118360f, 775.185227f, 12.610626f, 13.181156f, 0.867970f, 143.246243f, 3.296471f, 0.030564f, 763.594144f, 0.042535f, 0.328249f, 11.931409f, 159.057565f, 0.953864f, 421.181285f, 265.933405f, 6.665454f, 380.509874f, 29.711401f, 0.713783f, 17.999663f, 231.845135f, 0.085859f, 5.149207f, 9.788728f, 32421.029062f, 6.154606f, 4.690575f, 6.829275f, 2.617148f, 0.239970f, 4975.740974f, 26.622357f, 56.946994f, 102.047628f, 3.687037f, 256.066687f, 3306.136479f, 0.122345f, 1224.700907f, 6.349995f, 38.533871f, 0.853898f, 30.134523f, 0.972516f, 571.481225f, 1334.690528f, 2.629860f, 136.872240f, 4504.948219f, 26.717363f, 0.236227f, 528.564753f, 10.731213f, 119.897459f, 231.801699f, 2.303594f, 419.465390f, 4.480951f, 0.255509f, 3.705470f, 4.674319f, 21.151329f, 6.680614f, 12.912784f, 30.747629f, 48.900290f, 61.363645f, 136.659114f, 2945.164242f, 8.961418f, 29.622364f, 4.550997f, 0.893408f, 1.618886f, 122.930966f, 0.061735f, 4.421470f, 0.230336f, 301.764922f, 14.392640f, 355.803544f, 482.797761f, 5.815136f, 18.428734f, 4.699128f, 25.882393f, 2.205922f, 0.514218f, 103.862139f, 134.025758f, 6.139302f, 51.773859f, 2.383357f, 1.961648f, 29.210866f, 11.534690f, 10.700701f, 3.752292f, 8.451452f, 0.703204f, 6.069601f, 78.043936f, 26.800164f, 111.121891f, 29.390488f, 12.150925f, 322.146267f, 7.043491f, 37.251711f, 94.688547f, 0.085815f, 14.067185f, 568.165127f, 8486.846025f, 287.932107f, 874.829965f, 0.959394f, 3.705974f, 7.197836f, 44.761765f, 3240.291873f, 9.831951f, 0.851823f, 135.568151f, 1656.542100f, 7.755518f, 208.390612f, 55.363207f, 174.993103f, 300.724475f, 96.018774f, 0.784093f, 42.402202f, 0.220304f, 1.703785f, 50.265200f, 0.713332f, 2048.531089f, 147.998970f, 79.141670f, 58.725271f, 290.827415f, 10.350575f, 1678.495382f, 10.071883f, 614.892113f, 447.622323f, 0.702796f, 57.228797f, 17.759891f, 51.646707f, 0.131755f, 0.059067f, 1.916636f, 1.381680f, 20.384632f, 5.824009f, 19.398318f, 23.850457f, 203.379146f, 6.437796f, 1.743421f, 57.392472f, 8.774599f, 18.962306f, 370.721188f, 41.146514f, 435.977752f, 0.894587f, 3200.397611f
};

// Scaler constants (scaler_params.json), baked in as a real deployment would.
static const float SCALER_MEAN[3] = {0.4710637576573735f, 27.427364864864863f, 5.195225959771762f};
static const float SCALER_STD[3]  = {0.6341238117450851f, 14.3662209790266f, 0.7290963592377475f};

// Fix the CPU frequency so results are reproducible across boards/runs.
// 240 MHz is the max on ESP32; report whatever you actually pin here.
static const uint32_t TARGET_CPU_MHZ = 240;

volatile float sink = 0.0f;  // prevents dead-code elimination

float predict_from_raw(int idx) {
    float kn = pgm_read_float(&SAMPLE_KN[idx]);
    float ar = pgm_read_float(&SAMPLE_AR[idx]);
    float freq = pgm_read_float(&SAMPLE_FREQ[idx]);
    float skn = (kn - SCALER_MEAN[0]) / SCALER_STD[0];
    float sar = (ar - SCALER_MEAN[1]) / SCALER_STD[1];
    float logf = log10f(freq);
    float slf = (logf - SCALER_MEAN[2]) / SCALER_STD[2];
    return predict_log_damping(skn, sar, slf);
}

void setup() {
    Serial.begin(115200);
    delay(2000);  // let Serial Monitor connect

    // --- Reduce timing jitter sources ---
    WiFi.mode(WIFI_OFF);
    btStop();
    esp_bt_controller_disable();

    setCpuFrequencyMhz(TARGET_CPU_MHZ);
    delay(10);

    Serial.println("=== ESP32 edge_model.h inference benchmark ===");
    Serial.printf("Requested CPU freq: %u MHz | Actual: %u MHz\n",
                   TARGET_CPU_MHZ, getCpuFrequencyMhz());
    Serial.printf("N_SAMPLES: %d (real dataset rows, seed=42)\n\n", N_SAMPLES);

    // --- On-device correctness check, in the SAME binary that gets timed ---
    double sq_err_sum = 0.0;
    for (int i = 0; i < N_SAMPLES; ++i) {
        float pred_log10_cd = predict_from_raw(i);
        float true_cd = pgm_read_float(&SAMPLE_CD[i]);
        double true_log10_cd = log10((double)true_cd);
        double diff = (double)pred_log10_cd - true_log10_cd;
        sq_err_sum += diff * diff;
        sink += pred_log10_cd;
    }
    double mse = sq_err_sum / N_SAMPLES;
    Serial.printf("On-device log10-space MSE (this build, this sample): %.6f\n", mse);
    Serial.println("(compare to paper's edge.mse = 0.151565 on the 1,480-row held-out test set");
    Serial.println(" [20% split of the 7,400-row dataset]; this is a 200-row spot check on");
    Serial.println(" real rows, not a replacement for the full-test-set number.)\n");

    // --- Warmup ---
    const int WARMUP = 200;
    for (int i = 0; i < WARMUP; ++i) sink += predict_from_raw(i % N_SAMPLES);

    // --- 5 independent timed batches -> mean +/- std ---
    const int N_BATCHES = 5;
    const long CALLS_PER_BATCH = 50000;
    double batch_ns_per_call[N_BATCHES];

    for (int b = 0; b < N_BATCHES; ++b) {
        int64_t t0 = esp_timer_get_time();  // microseconds
        for (long i = 0; i < CALLS_PER_BATCH; ++i) {
            sink += predict_from_raw(i % N_SAMPLES);
        }
        int64_t t1 = esp_timer_get_time();
        double total_us = (double)(t1 - t0);
        batch_ns_per_call[b] = (total_us * 1000.0) / CALLS_PER_BATCH;
        Serial.printf("Batch %d/%d: %.3f us/call (%ld calls, %.2f ms total)\n",
                      b + 1, N_BATCHES, batch_ns_per_call[b] / 1000.0,
                      CALLS_PER_BATCH, total_us / 1000.0);
        delay(50);  // brief gap between batches
    }

    double sum = 0;
    for (int b = 0; b < N_BATCHES; ++b) sum += batch_ns_per_call[b];
    double mean_ns = sum / N_BATCHES;
    double sq = 0;
    for (int b = 0; b < N_BATCHES; ++b) sq += (batch_ns_per_call[b] - mean_ns) * (batch_ns_per_call[b] - mean_ns);
    double std_ns = sqrt(sq / N_BATCHES);

    Serial.printf("\n=== RESULT: %.3f +/- %.3f us/call (n=%d batches, %ld calls each) ===\n",
                  mean_ns / 1000.0, std_ns / 1000.0, N_BATCHES, CALLS_PER_BATCH);
    Serial.printf("Throughput: %.0f calls/sec\n", 1e9 / mean_ns);
    Serial.printf("CPU freq during test: %u MHz | WiFi/BT: off\n", getCpuFrequencyMhz());

    Serial.printf("\n(sink=%f -- ignore, prevents optimizer from deleting the calls)\n", sink);
    Serial.println("\nThis number is real hardware timing -- safe to cite in Sec. 7.3 as measured,");
    Serial.println("not theoretical. Report it as mean +/- std with n=5 batches, this board model,");
    Serial.println("and the pinned CPU frequency above.");
}

void loop() {
    // nothing -- all work happens once in setup()
}
