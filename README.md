from AlgorithmImports import *
import random
import pandas as pd
import numpy as np
import tensorflow as tf

# Configure TensorFlow GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Only allocate memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Alternatively, limit memory usage to a specific amount
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
        # )
        print(f"GPU memory growth set to dynamic for {len(gpus)} GPUs")
    except RuntimeError as e:
        print(f"GPU memory configuration error: {e}")
else:
    print("No GPUs detected, falling back to CPU")

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))
try:
    print("CUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
    print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
except:
    print("CUDA/cuDNN version info not available")
import sys
print("Python Executable: " + sys.executable)
print("TensorFlow version: " + tf.__version__)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Lambda, LSTM, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from datetime import timedelta
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
from sklearn.impute import SimpleImputer
import tensorflow.keras.layers as layers
from collections import deque

BATCH_SIZE = 128
VAE_TRAIN_INTERVAL = 60
USE_NUMPY = True
VECTORIZE_FEATURES = True
MAX_BUFFER_SIZE = 5000
TFT_HIDDEN_DIM = 32
TFT_NUM_HEADS = 1
TFT_NUM_BLOCKS = 1

VAE_LOSS_WINDOW = 10
RETRAIN_THRESHOLD = 1.1

PPO_TRAIN_FREQUENCY = 100

def batch_predict(model, batch_data):
    batch_data = np.array(batch_data)
    return model.predict(batch_data, batch_size=BATCH_SIZE)

def ema_normalize(data, alpha=0.1):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t-1]
    ema_var = np.zeros_like(data)
    ema_var[0] = 0
    for t in range(1, len(data)):
        ema_var[t] = alpha * (data[t] - ema[t])**2 + (1 - alpha) * ema_var[t-1]
    std = np.sqrt(ema_var)
    return (data - ema) / (std + 1e-8)

def compute_slippage(volume, liquidity):
    ratio = volume / (liquidity + 1e-8)
    base_slippage = 0.001
    return base_slippage * (1 + ratio)

def build_lstm_encoder(input_dim, lstm_units=64, dense_units=32):
    inputs = Input(shape=(None, input_dim))
    x = LSTM(lstm_units, return_sequences=True)(inputs)
    attention = Attention()([x, x])
    x = LSTM(lstm_units // 2, return_sequences=False)(attention)
    x = Dense(dense_units, activation='relu')(x)
    return Model(inputs, x, name="LSTM_Encoder")

latent_dim = 8
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    epsilon = tf.random.normal(shape=(batch, latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim):
    inputs = Input(shape=(input_dim,))
    h = Dense(32, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="VAE_Encoder")
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(32, activation='relu')(latent_inputs)
    outputs = Dense(input_dim, activation='linear')(x)
    decoder = Model(latent_inputs, outputs, name="VAE_Decoder")
    return encoder, decoder

class CombinedVAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    @property
    def metrics(self):
        return [self.loss_tracker]
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(data, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + kl_loss
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

def build_tft(input_dim, output_dim=3, hidden_dim=TFT_HIDDEN_DIM, num_heads=TFT_NUM_HEADS, num_blocks=TFT_NUM_BLOCKS, dropout_rate=0.1):
    inputs = Input(shape=(None, input_dim))
    variable_selection = Dense(hidden_dim, activation="relu")(inputs)
    variable_selection = Dense(input_dim, activation="sigmoid")(variable_selection)
    x = layers.multiply([inputs, variable_selection])
    x = Dense(hidden_dim)(x)
    for _ in range(num_blocks):
        attn_output = layers.Attention()([x, x])
        attn_output = layers.LayerNormalization()(attn_output + x)
        x = Dense(hidden_dim, activation="relu")(attn_output)
        x = layers.LayerNormalization()(x + attn_output)
    outputs = Dense(output_dim, activation="softmax")(x[:, -1, :])
    return Model(inputs, outputs, name="TFT_Trade_Classifier")

def build_tft_forecast(input_dim, output_dim=1):
    inputs = Input(shape=(None, input_dim))
    x = Dense(32, activation="relu")(inputs)
    for _ in range(2):
        attn_output = Attention()([x, x])
        x = Dense(32, activation="relu")(attn_output)
    outputs = Dense(output_dim)(x[:, -1, :])
    return Model(inputs, outputs, name="TFT_Forecaster")

def create_sequences(data, seq_length):
    sequences = []
    for i in range(0, len(data) - seq_length + 1):
         sequences.append(data[i:i+seq_length])
    return np.array(sequences)

def compute_atr(prices, period=14):
    df = pd.DataFrame({"high": prices, "low": prices, "close": prices})
    df["tr"] = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - df["close"].shift()), abs(df["low"] - df["close"].shift())))
    return df["tr"].rolling(window=period).mean().iloc[-1]

def tft_uncertainty_predict(tft_model, input_data, num_samples=50):
    predictions = np.array([tft_model(input_data, training=True).numpy() for _ in range(num_samples)])
    mean_prediction = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)
    return mean_prediction, uncertainty

def compute_kelly_fraction(win_prob, risk_reward_ratio):
    return win_prob - ((1 - win_prob) / risk_reward_ratio)

class ExperienceReplay:
    def __init__(self, capacity=5000):
        self.memory = deque(maxlen=capacity)
    def add(self, experience):
        self.memory.append(experience)
    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

class PPOAgent:
    def __init__(self, state_dim, action_dim, tft_model=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tft = tft_model
        self.actor, self.critic = self.build_models()
        self.optimizer = Adam(learning_rate=0.0003)
        self.epsilon_clip = 0.2
        self.gamma = 0.99
        self.buffer = []
    def build_models(self):
        state_input = Input(shape=(self.state_dim,))
        dense1 = Dense(64, activation='relu')(state_input)
        dense2 = Dense(64, activation='relu')(dense1)
        out_actions = Dense(self.action_dim, activation='softmax')(dense2)
        actor = Model(state_input, out_actions)
        dense1_c = Dense(64, activation='relu')(state_input)
        dense2_c = Dense(64, activation='relu')(dense1_c)
        out_value = Dense(1, activation='linear')(dense2_c)
        critic = Model(state_input, out_value)
        return actor, critic

    def act(self, state):
        state_seq = state.reshape(1, 1, -1)
        mean_prediction, _ = tft_uncertainty_predict(self.tft, state_seq, num_samples=50)
        # Check for NaNs or zero sum in the probability vector
        if np.any(np.isnan(mean_prediction)):
            print("DEBUG: NaN detected in TFT output, using uniform distribution.")
            mean_prediction = np.ones((1, self.action_dim)) / self.action_dim
        p = mean_prediction[0]
        total = np.sum(p)
        if total == 0:
            print("DEBUG: Sum of probabilities is zero, using uniform distribution.")
            p = np.ones(self.action_dim) / self.action_dim
        else:
            p = p / total
        action = np.random.choice(self.action_dim, p=p)
        return action, p[action]

    def remember(self, state, action, reward, next_state, done, old_prob):
        self.buffer.append((state, action, reward, next_state, done, old_prob))
    def train(self, epochs=5, batch_size=16):
        if len(self.buffer) < batch_size:
            return
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones, old_probs = map(np.array, zip(*[self.buffer[i] for i in indices]))
        discounted_rewards = []
        R = 0
        for r, done in zip(rewards[::-1], dones[::-1]):
            if done:
                R = 0
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                probs = self.actor(states)
                indices_tensor = tf.range(batch_size)
                action_indices = tf.stack([indices_tensor, actions], axis=1)
                new_probs = tf.gather_nd(probs, action_indices)
                ratio = new_probs / (old_probs + 1e-10)
                values = tf.squeeze(self.critic(states))
                advantages = discounted_rewards - values
                advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                critic_loss = tf.reduce_mean(tf.square(advantages))
                loss = actor_loss + 0.5 * critic_loss
            grads = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))
        self.buffer = []

class AnomalyTrainingAlgorithm(QCAlgorithm):
    def Initialize(self):
        print("Starting to initialize algorithm...")
        print("TensorFlow version:", tf.__version__)
        print("GPUs available:", tf.config.list_physical_devices('GPU'))
        
        self.SetStartDate(2023, 12, 7)
        self.SetEndDate(2024, 5, 1)
        self.SetCash(10000)
        self.settings.daily_precise_end_time = False
        # Change market to BYBIT to match your downloaded data
        self.symbol = self.AddCrypto("BTCUSDT", Resolution.Minute, Market.BYBIT).Symbol
        
        print(f"Added symbol: {self.symbol} from market {Market.BYBIT}")
        
        self.state = "TRAINING"
        self.dataPullingStart = self.Time
        self.lastTrainingRestart = self.Time
        self.rawData = []
        self.fractional_diff_order = 0.4
        self.ofi_window = 30
        self.extra_window = 20
        
        # Initialize TensorFlow models
        print("Initializing TensorFlow models...")
        self.lstm_encoder = build_lstm_encoder(input_dim=8, lstm_units=64, dense_units=32)
        self.lstm_encoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        self.vae_encoder, self.vae_decoder = build_vae(input_dim=32)
        self.tft = build_tft(input_dim=8, output_dim=3, hidden_dim=TFT_HIDDEN_DIM, num_heads=TFT_NUM_HEADS, num_blocks=TFT_NUM_BLOCKS)
        self.combined_vae = CombinedVAE(self.vae_encoder, self.vae_decoder)
        self.combined_vae.compile(optimizer=Adam(learning_rate=0.0005))
        self.rlAgent = PPOAgent(state_dim=8, action_dim=3, tft_model=self.tft)
        
        print("Models initialized successfully")
        
        self.tradeHistory = []
        self.tradeReturns = []
        self.activeTrade = None
        self.portfolioPeak = self.Portfolio.TotalPortfolioValue
        self.feature_threshold = None
        self.daily_start_equity = self.Portfolio.TotalPortfolioValue
        self.lastVaeTraining = self.Time
        self.vae_loss_history = deque(maxlen=VAE_LOSS_WINDOW)
        self.previous_embeddings = None
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(23,55), self.DailyMetricsReport)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0,0), self.UpdateState)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0,5), self.CheckForTrainingRestart)
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday), self.TimeRules.At(0,5), self.RetrainTFT)

    def DailyMetricsReport(self):
        cumulative_pnl = self.Portfolio.TotalPortfolioValue - self.daily_start_equity
        if len(self.tradeHistory) > 0:
            win_rate = sum(label for (_, label) in self.tradeHistory) / len(self.tradeHistory)
            avg_return = np.mean(self.tradeReturns)
        else:
            win_rate = 0
            avg_return = 0
        self.Debug("Daily Metrics Report - Model State: " + self.state + " | Cumulative PnL: " + str(cumulative_pnl) + " | Avg Win Rate: " + str(win_rate) + " | Avg Return/Trade: " + str(avg_return))
        self.daily_start_equity = self.Portfolio.TotalPortfolioValue
        self.tradeHistory = []
        self.tradeReturns = []

    def UpdateState(self):
        elapsed = self.Time - self.dataPullingStart
        self.state = "TRAINING" if elapsed < timedelta(days=1) else "TRADING"

    def CheckForTrainingRestart(self):
        if self.Time - self.lastTrainingRestart >= timedelta(days=1) or self.DetectDrift():
            self.lastTrainingRestart = self.Time
            self.TrainingPhaseTasks()

    def DetectDrift(self):
        if len(self.rawData) < 150:
            return False
        if USE_NUMPY:
            prices = np.array([d["price"] for d in self.rawData[-150:]])
            drift_signal = abs(prices[-20:].mean() - prices[-100:].mean()) / (prices[-100:].mean() + 1e-8)
        else:
            df = pd.DataFrame(self.rawData[-150:])
            short_rolling = df["price"].rolling(window=20).mean()
            long_rolling = df["price"].rolling(window=100).mean()
            drift_signal = abs(short_rolling.iloc[-1] - long_rolling.iloc[-1]) / (long_rolling.iloc[-1] + 1e-8)
        if not hasattr(self, "drift_ema"):
            self.drift_ema = drift_signal
        else:
            alpha = 0.1
            self.drift_ema = alpha * drift_signal + (1 - alpha) * self.drift_ema
        self.Debug("Drift EMA: " + str(self.drift_ema))
        return self.drift_ema > 0.05

    def GetAggregatedFeatureSequence(self):
        if len(self.rawData) < 60:
            return None
        if USE_NUMPY:
            data = self.rawData[-60:]
            price = np.array([d["price"] for d in data])
            bidVolume = np.array([d["bidVolume"] for d in data])
            askVolume = np.array([d["askVolume"] for d in data])
            bidAskSpread = np.array([d["bidAskSpread"] for d in data])
            volume = np.array([d["volume"] for d in data])
            orderValue = np.array([d["orderValue"] for d in data])
            liquidity = np.array([d["liquidity"] for d in data])
            if VECTORIZE_FEATURES:
                price_frac_diff = self.FractionalDifferentiation(price, self.fractional_diff_order)
            else:
                price_frac_diff = self.FractionalDifferentiation(price, self.fractional_diff_order)
            seg_length = 10
            seq = []
            for i in range(0, 60, seg_length):
                seg_price = price[i:i+seg_length]
                seg_bidVolume = bidVolume[i:i+seg_length]
                seg_askVolume = askVolume[i:i+seg_length]
                seg_bidAskSpread = bidAskSpread[i:i+seg_length]
                seg_volume = volume[i:i+seg_length]
                seg_orderValue = orderValue[i:i+seg_length]
                seg_liquidity = liquidity[i:i+seg_length]
                seg_price_frac_diff = price_frac_diff[i:i+seg_length]
                agg = [np.nanmean(seg_price), np.nanmean(seg_price_frac_diff), np.nansum(seg_bidVolume), np.nansum(seg_askVolume), np.nanmean(seg_bidAskSpread), np.nansum(seg_volume), np.nansum(seg_orderValue), np.nansum(seg_liquidity)]
                seq.append(agg)
            return np.array([seq])
        else:
            df = pd.DataFrame(self.rawData[-60:])
            if VECTORIZE_FEATURES:
                df["price_frac_diff"] = np.vectorize(lambda x: self.FractionalDifferentiation(x, self.fractional_diff_order))(df["price"])
            else:
                df["price_frac_diff"] = self.FractionalDifferentiation(df["price"], self.fractional_diff_order)
            df["roll_spread"] = self.ComputeRollSpread(df["price"])
            df["kyle_lambda"] = self.ComputeKyleLambda(df["volume"], df["price"])
            df["hasbrouck"] = self.ComputeHasbrouckImpact(df["price"], df["volume"])
            df["ofi"] = df["bidVolume"].rolling(window=self.ofi_window, min_periods=1).sum() - df["askVolume"].rolling(window=self.ofi_window, min_periods=1).sum()
            keys = ["price", "price_frac_diff", "bidVolume", "askVolume", "bidAskSpread", "volume", "orderValue", "liquidity", "roll_spread", "kyle_lambda", "hasbrouck", "ofi"]
            seq = []
            for i in range(0, 60, 10):
                segment = df.iloc[i:i+10]
                agg_features = [segment[k].mean() if k in ["price", "price_frac_diff", "bidAskSpread", "roll_spread", "kyle_lambda", "hasbrouck", "ofi"] else segment[k].sum() for k in keys]
                seq.append(agg_features)
            return np.array([seq])

    def AggregateAndTrainModel(self):
        try:
            window_days = 20
            filtered_data = [dp for dp in self.rawData if dp["time"] >= self.Time - timedelta(days=window_days)]
            if len(filtered_data) < 60:
                return
            training_samples = []
            window_size = 60
            for i in range(0, len(filtered_data) - window_size + 1, 10):
                window_data = filtered_data[i:i+window_size]
                fv = self.ComputeFeatureVectorFromWindow(window_data)
                training_samples.append(fv)
            if len(training_samples) < 10:
                return
            training_data = np.array(training_samples)
            normalized_data = ema_normalize(training_data, alpha=0.1)
            imputer = SimpleImputer(strategy='mean')
            normalized_data = imputer.fit_transform(normalized_data)
            sequences = create_sequences(normalized_data, seq_length=6)
            if sequences.shape[0] < 1:
                return
            lstm_features = batch_predict(self.lstm_encoder, sequences)
            if self.previous_embeddings is not None:
                divergence = self.compute_kl_divergence(self.previous_embeddings, lstm_features)
                if divergence > 0.05:
                    self.lstm_encoder.fit(sequences, epochs=5, batch_size=32, verbose=0)
            self.previous_embeddings = lstm_features
            reconstructions = batch_predict(self.combined_vae, lstm_features)
            current_loss = np.mean(np.square(lstm_features - reconstructions))
            self.vae_loss_history.append(current_loss)
            if self.should_retrain_vae(current_loss):
                self.combined_vae.fit(lstm_features, epochs=10, batch_size=16, verbose=0)
                self.lastVaeTraining = self.Time
            errors = np.mean(np.square(lstm_features - reconstructions), axis=1)
            self.feature_threshold = np.percentile(errors, 97)
            self.last_training_error = np.mean(errors)
            self.Debug("Training metrics - LSTM error: " + str(self.last_training_error) + " | Threshold: " + str(self.feature_threshold))
        except Exception as e:
            self.Debug(str(e))

    def should_retrain_vae(self, current_loss):
        if len(self.vae_loss_history) < VAE_LOSS_WINDOW:
            return False
        avg_past_loss = np.mean(self.vae_loss_history)
        return current_loss > avg_past_loss * RETRAIN_THRESHOLD

    def compute_kl_divergence(self, old_embeddings, new_embeddings):
        from scipy.stats import entropy
        old_hist, _ = np.histogram(old_embeddings, bins=50, density=True)
        new_hist, _ = np.histogram(new_embeddings, bins=50, density=True)
        return entropy(old_hist + 1e-8, new_hist + 1e-8)

    def ComputeFeatureVectorFromWindow(self, window_data):
        if USE_NUMPY:
            data = window_data
            price = np.array([d["price"] for d in data])
            bidVolume = np.array([d["bidVolume"] for d in data])
            askVolume = np.array([d["askVolume"] for d in data])
            bidAskSpread = np.array([d["bidAskSpread"] for d in data])
            volume = np.array([d["volume"] for d in data])
            orderValue = np.array([d["orderValue"] for d in data])
            liquidity = np.array([d["liquidity"] for d in data])
            if VECTORIZE_FEATURES:
                price_frac_diff = self.FractionalDifferentiation(price, self.fractional_diff_order)
            else:
                price_frac_diff = self.FractionalDifferentiation(price, self.fractional_diff_order)
            agg = [np.nanmean(price), np.nanmean(price_frac_diff), np.nansum(bidVolume), np.nansum(askVolume), np.nanmean(bidAskSpread), np.nansum(volume), np.nansum(orderValue), np.nansum(liquidity)]
            return agg
        else:
            df = pd.DataFrame(window_data)
            df["price_frac_diff"] = self.FractionalDifferentiation(df["price"], self.fractional_diff_order)
            keys = ["price", "price_frac_diff", "bidVolume", "askVolume", "bidAskSpread", "volume", "orderValue", "liquidity"]
            return [df[k].mean() if k in ["price", "price_frac_diff", "bidAskSpread"] else df[k].sum() for k in keys]

    def FractionalDifferentiation(self, series, d, thresh=0.01):
        T = len(series)
        weights = [1.0]
        for k in range(1, T):
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < thresh:
                break
            weights.append(w)
        weights = np.array(weights[::-1]).reshape(-1, 1)
        diff_series = np.convolve(series, weights.flatten(), mode='valid')
        return np.concatenate((np.full(len(series) - len(diff_series), np.nan), diff_series))

    def ComputeAdditionalFeatures(self, df):
        df["vpin"] = (df["askVolume"] - df["bidVolume"]).abs() / (df["askVolume"] + df["bidVolume"] + 1e-8)
        df["wpc"] = df["price"] * df["volume"] / (df["volume"].sum() + 1e-8)
        df["tv_imbalance"] = df["volume"].diff().fillna(0)
        df["price_impact"] = df["price"].diff().abs() / (df["volume"].diff().abs() + 1e-8)
        return df

    def ComputeRollSpread(self, price_series):
        diff = price_series.diff().dropna() if not USE_NUMPY else np.diff(price_series)
        cov = diff.autocorr(lag=1) if not USE_NUMPY else np.corrcoef(diff[:-1], diff[1:])[0,1]
        return 2 * np.sqrt(-cov) if cov < 0 else 0.0

    def ComputeKyleLambda(self, volume_series, price_series):
        if USE_NUMPY:
            price_diff = np.diff(price_series)
            vol = np.std(price_diff)
            avg_volume = np.mean(volume_series)
        else:
            price_diff = price_series.diff().dropna()
            vol = price_diff.std()
            avg_volume = volume_series.mean()
        return vol / avg_volume if avg_volume != 0 else 0.0

    def ComputeHasbrouckImpact(self, price_series, volume_series):
        if USE_NUMPY:
            vol = np.std(price_series)
            avg_volume = np.mean(volume_series)
        else:
            vol = price_series.std()
            avg_volume = price_series.mean()
        return vol / avg_volume if avg_volume != 0 else 0.0

    def RecordTradeOutcome(self, exitPrice):
        if self.activeTrade is None:
            return
        direction = self.activeTrade["direction"]
        entryPrice = self.activeTrade["entryPrice"]
        profit = exitPrice - entryPrice if direction == -1 else entryPrice - exitPrice
        label = 1 if profit > 0 else 0
        investedCapital = entryPrice * abs(self.activeTrade["quantity"])
        tradeReturn = profit / investedCapital if investedCapital > 0 else 0
        self.tradeHistory.append((self.activeTrade["latent"], label))
        self.tradeReturns.append(tradeReturn)

    def ComputeDynamicTradeFraction(self, win_prob):
        risk_reward_ratio = 2.0
        kelly_fraction = compute_kelly_fraction(win_prob, risk_reward_ratio)
        return max(0.01, min(kelly_fraction, 0.3))

    def TrainingPhaseTasks(self):
        self.AggregateAndTrainModel()

    def compute_new_features(self, data):
        df = pd.DataFrame(data)
        df["lag_1d"] = df["price"].shift(1)
        df["lag_3d"] = df["price"].shift(3)
        df["lag_7d"] = df["price"].shift(7)
        df["ma_10"] = df["price"].rolling(window=10).mean()
        df["ma_30"] = df["price"].rolling(window=30).mean()
        df["atr"] = compute_atr(df["price"])
        df["ofi"] = df["bidVolume"] - df["askVolume"]
        return df.dropna().values

    def OnData(self, data):
        if not data.ContainsKey(self.symbol):
            return
        # Get bar data from trade and quote
        tradeBar = data.Bars.get(self.symbol, None)
        quoteBar = data.QuoteBars.get(self.symbol, None)
        if tradeBar is not None:
            price = tradeBar.Close
            volume = tradeBar.Volume
        else:
            price = None
            volume = 0
        if quoteBar is not None:
            bidVolume = quoteBar.Bid.Volume if hasattr(quoteBar.Bid, "Volume") else 0
            askVolume = quoteBar.Ask.Volume if hasattr(quoteBar.Ask, "Volume") else 0
            bidAskSpread = quoteBar.Ask.Close - quoteBar.Bid.Close
        else:
            bidVolume = 0
            askVolume = 0
            bidAskSpread = 0
        # Ensure price is valid before proceeding
        if price is None or price <= 0:
            return
        orderValue = price * volume
        liquidity = orderValue
        orderSize = (bidVolume + askVolume) / 2 if quoteBar is not None else volume
        dp = {"price": price, "bidVolume": bidVolume, "askVolume": askVolume, "bidAskSpread": bidAskSpread, "volume": volume, "orderValue": orderValue, "liquidity": liquidity, "orderSize": orderSize, "time": self.Time}
        self.rawData.append(dp)
        if self.state == "TRADING" and self.lstm_encoder is not None:
            # Only process on certain time steps
            if self.Time.minute % 5 != 0 and self.activeTrade is None:
                return
            seq = self.GetAggregatedFeatureSequence()
            if seq is None:
                return
            lstm_features = batch_predict(self.lstm_encoder, seq)
            z_mean, z_log_var, latent = batch_predict(self.vae_encoder, lstm_features)
            recon = batch_predict(self.vae_decoder, latent)
            error = np.mean((lstm_features - recon)**2)
            if self.feature_threshold is None:
                self.feature_threshold = error
            ae_signal = -1 if error > self.feature_threshold else 1
            action, action_prob = self.rlAgent.act(latent[0])
            atr_value = compute_atr([d["price"] for d in self.rawData[-14:]])
            # If in an active trade, manage it
            if self.activeTrade is not None and price is not None:
                if action == 1:
                    self.RecordTradeOutcome(price)
                    self.Liquidate(self.symbol)
                    self.activeTrade = None
                    return
                elif action == 2:
                    if self.activeTrade["direction"] == -1:
                        self.activeTrade["stopLoss"] = price + atr_value
                        self.activeTrade["takeProfit"] = price - atr_value
                    else:
                        self.activeTrade["stopLoss"] = price - atr_value
                        self.activeTrade["takeProfit"] = price + atr_value
                if self.activeTrade["direction"] == -1:
                    if price >= self.activeTrade["stopLoss"] or price <= self.activeTrade["takeProfit"]:
                        self.RecordTradeOutcome(price)
                        self.Liquidate(self.symbol)
                        self.activeTrade = None
                        return
                    if self.Time - self.activeTrade["entryTime"] >= timedelta(days=5):
                        self.RecordTradeOutcome(price)
                        self.Liquidate(self.symbol)
                        self.activeTrade = None
                        return
                elif self.activeTrade["direction"] == 1:
                    if price <= self.activeTrade["stopLoss"] or price >= self.activeTrade["takeProfit"]:
                        self.RecordTradeOutcome(price)
                        self.Liquidate(self.symbol)
                        self.activeTrade = None
                        return
                    if self.Time - self.activeTrade["entryTime"] >= timedelta(days=5):
                        self.RecordTradeOutcome(price)
                        self.Liquidate(self.symbol)
                        self.activeTrade = None
                        return
                next_state = latent[0]
                reward = (self.Portfolio.TotalPortfolioValue - self.Portfolio.TotalPortfolioValue) / self.Portfolio.TotalPortfolioValue
                self.rlAgent.remember(latent[0], action, reward, next_state, False, action_prob)
                if len(self.rlAgent.buffer) >= PPO_TRAIN_FREQUENCY:
                    self.rlAgent.train(epochs=5, batch_size=16)
            else:
                # Ensure we have valid price before computing order quantity
                if price is None or price <= 0:
                    return
                THRESHOLD = 0.6
                win_prob = action_prob
                tradeFraction = self.ComputeDynamicTradeFraction(win_prob)
                # Compute order quantity only if price is valid
                quantity = self.CalculateOrderQuantity(self.symbol, tradeFraction)
                quantity = max(quantity, 0.005)
                _ = compute_slippage(volume, liquidity)
                if ae_signal == 1 and win_prob >= THRESHOLD:
                    if price is None:
                        return
                    stopLoss = price - atr_value
                    takeProfit = price + atr_value
                    self.MarketOrder(self.symbol, quantity)
                    self.activeTrade = {"direction": 1, "entryPrice": price, "entryTime": self.Time, "stopLoss": stopLoss, "takeProfit": takeProfit, "latent": latent[0], "quantity": quantity}
                elif ae_signal == -1 and win_prob >= THRESHOLD:
                    if price is None:
                        return
                    stopLoss = price + atr_value
                    takeProfit = price - atr_value
                    self.MarketOrder(self.symbol, -quantity)
                    self.activeTrade = {"direction": -1, "entryPrice": price, "entryTime": self.Time, "stopLoss": stopLoss, "takeProfit": takeProfit, "latent": latent[0], "quantity": -quantity}

    def OnEndOfAlgorithm(self):
        pass

    def RetrainTFT(self):
        if self.DetectDrift():
            self.tft.fit(self.tft_training_data, self.tft_labels, epochs=10, batch_size=32, verbose=0)
