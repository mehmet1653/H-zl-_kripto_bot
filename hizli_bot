import os
import time
import threading
import sys
import pickle
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from dotenv import load_dotenv

# Kendi modüllerimiz
from indicators import puanla
import db

load_dotenv()

sys.stdout.reconfigure(line_buffering=True)
app = Flask(__name__)

# ==================== AYARLAR ====================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")
GATE_API_KEY = os.environ.get("GATE_API_KEY", "")
GATE_SECRET = os.environ.get("GATE_SECRET", "")

# Gate.io Testnet
exchange = ccxt.gate({
    'apiKey': GATE_API_KEY,
    'secret': GATE_SECRET,
    'enableRateLimit': True,
    'timeout': 20000,
    'options': {'defaultType': 'swap'}
})
exchange.set_sandbox_mode(True)

# ==================== ML MODELİ YÜKLE ====================
MODEL_DOSYA = 'models/best_model.pkl'
ML_MODEL = None
ML_OZELLIKLER = []
ML_ESIK = 0.5

try:
    with open(MODEL_DOSYA, 'rb') as f:
        model_data = pickle.load(f)
        ML_MODEL = model_data['model']
        ML_OZELLIKLER = model_data['ozellikler']
        ML_ESIK = model_data.get('esik', 0.5)
        print(f"✅ ML modeli yüklendi: {model_data.get('model_adi', '?')}, eşik: {ML_ESIK}")
        print(f"   Özellikler ({len(ML_OZELLIKLER)}): {ML_OZELLIKLER}")
except Exception as e:
    print(f"❌ ML modeli yüklenemedi: {e}")
    ML_MODEL = None

# ==================== SABITLER ====================
COINLER = ['ADA', 'AVAX', 'DOGE', 'LINK', 'PEPE', 'SOL', 'SUI', 'TON', 'WIF', 'XRP']
ZAMAN_DILIMI = '4h'
MIN_PUAN = 50
MAKS_POZISYON = 3
KOMISYON = 0.0005
SLIPPAGE = 0.0003
TOPLAM_MALIYET = (KOMISYON + SLIPPAGE) * 2

# ==================== DURUMLAR ====================
BOT_CALISIYOR_MU = True
AKTIF_POZISYONLAR = {}
COIN_COOLDOWNLAR = {}
SON_MUM_TS = {}

# ==================== TELEGRAM ====================
def telegram_mesaj_gonder(mesaj):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print(f"📨 [TG-KAPALI] {mesaj}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": mesaj, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        print(f"📨 [TG HATA] {e}")

# ==================== ML FİLTRE ====================
def ml_onayla(ozellikler_dict):
    """ML modelinden onay al."""
    if ML_MODEL is None:
        return True, 0.5
    
    try:
        vektor = []
        for oz in ML_OZELLIKLER:
            v = ozellikler_dict.get(oz, 0)
            if pd.isna(v) or np.isinf(v):
                v = 0
            vektor.append(float(v))
        
        X = np.array([vektor])
        proba = ML_MODEL.predict_proba(X)[0, 1]
        
        return bool(proba >= ML_ESIK), float(proba)
    except Exception as e:
        print(f"⚠️ ML hata: {e}")
        return True, 0.5

# ==================== CANLI VERİ ====================
def canli_mumlari_cek(symbol, limit=200):
    for deneme in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, ZAMAN_DILIMI, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except Exception as e:
            print(f"⚠️ {symbol} veri (deneme {deneme+1}): {str(e)[:50]}")
            time.sleep(2)
    return None

# ==================== ÖZELLİK ÇIKARMA ====================
def canli_ozellikleri_cikar(df):
    from indicators import (
        rsi_hesapla, bollinger_hesapla, macd_hesapla,
        atr_hesapla, adx_hesapla, ema_hesapla, hacim_orani
    )
    
    if len(df) < 60:
        return None
    
    try:
        close = df['close']
        
        rsi = rsi_hesapla(close).iloc[-1]
        sma, ust, alt = bollinger_hesapla(close)
        son_ust = ust.iloc[-1]
        son_alt = alt.iloc[-1]
        son_fiyat = close.iloc[-1]
        
        macd_line, signal_line, hist = macd_hesapla(close)
        son_hist = hist.iloc[-1]
        onceki_hist = hist.iloc[-2]
        
        atr = atr_hesapla(df).iloc[-1]
        adx = adx_hesapla(df).iloc[-1]
        ema20 = ema_hesapla(close, 20).iloc[-1]
        ema50 = ema_hesapla(close, 50).iloc[-1]
        ema200 = ema_hesapla(close, 200).iloc[-1] if len(close) >= 200 else ema50
        hacim_or = hacim_orani(df)
        
        atr_pct = (atr / son_fiyat) * 100 if son_fiyat > 0 else 0
        bollinger_pozisyon = (son_fiyat - son_alt) / (son_ust - son_alt) if (son_ust - son_alt) > 0 else 0.5
        
        son_5_getiri = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100 if len(close) >= 5 else 0
        son_10_getiri = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] * 100 if len(close) >= 10 else 0
        son_20_getiri = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100 if len(close) >= 20 else 0
        son_20_std = close.pct_change().rolling(20).std().iloc[-1] * 100 if len(close) >= 20 else 0
        
        return {
            'rsi': float(rsi),
            'bollinger_pozisyon': float(bollinger_pozisyon),
            'macd_hist': float(son_hist),
            'macd_hist_degisim': float(son_hist - onceki_hist),
            'atr_pct': float(atr_pct),
            'adx': float(adx),
            'ema20_50_fark': float((ema20 - ema50) / ema50 * 100 if ema50 > 0 else 0),
            'ema50_200_fark': float((ema50 - ema200) / ema200 * 100 if ema200 > 0 else 0),
            'fiyat_ema20_pct': float((son_fiyat - ema20) / ema20 * 100 if ema20 > 0 else 0),
            'fiyat_ema50_pct': float((son_fiyat - ema50) / ema50 * 100 if ema50 > 0 else 0),
            'hacim_orani': float(hacim_or),
            'son_5_getiri': float(son_5_getiri),
            'son_10_getiri': float(son_10_getiri),
            'son_20_getiri': float(son_20_getiri),
            'son_20_std': float(son_20_std),
        }
    except Exception as e:
        print(f"⚠️ Özellik hatası: {e}")
        return None

# ==================== KALDIRAÇ VE KASA ====================
def kaldirac_ve_kasa(puan):
    if puan >= 100:
        return 10, 0.20
    elif puan >= 85:
        return 5, 0.10
    elif puan >= 80:
        return 3, 0.05
    elif puan >= 75:
        return 2, 0.03
    elif puan >= MIN_PUAN:
        return 2, 0.03
    return 0, 0

# ==================== POZİSYON AÇMA ====================
def pozisyon_ac(symbol, sig):
    try:
        bal = exchange.fetch_balance()
        kasa = float(bal['total'].get('USDT', 0))
        if kasa < 10:
            print(f"⚠️ Kasa yetersiz: {kasa}")
            return False
        
        kaldirac, kasa_pct = kaldirac_ve_kasa(sig['puan'])
        try:
            exchange.set_leverage(kaldirac, symbol)
        except Exception:
            pass
        
        risk_usdt = kasa * kasa_pct
        stop_pct = sig['stop_pct']
        poz_degeri = risk_usdt / stop_pct if stop_pct > 0 else risk_usdt
        
        market_info = exchange.market(symbol)
        cs = float(market_info.get('contractSize', 1.0))
        ham = poz_degeri / (sig['fiyat'] * cs)
        miktar = float(exchange.amount_to_precision(symbol, max(ham, 0.001)))
        
        if miktar <= 0:
            return False
        
        yon = 'buy' if sig['yon'] == 'LONG' else 'sell'
        emir = exchange.create_order(symbol, 'market', yon, miktar)
        giris = float(emir.get('average') or emir.get('price') or sig['fiyat'])
        time.sleep(0.5)
        
        if sig['yon'] == 'LONG':
            stop = giris * (1 - stop_pct)
            tp = giris * (1 + sig['tp_pct'])
            kapat_yon = 'sell'
        else:
            stop = giris * (1 + stop_pct)
            tp = giris * (1 - sig['tp_pct'])
            kapat_yon = 'buy'
        
        stop = float(exchange.price_to_precision(symbol, stop))
        tp = float(exchange.price_to_precision(symbol, tp))
        
        try:
            exchange.create_order(
                symbol, 'stop', kapat_yon, miktar, stop,
                {'stopPrice': stop, 'triggerPrice': stop, 'reduceOnly': True}
            )
        except Exception as e:
            print(f"⚠️ Stop emri: {e}")
        
        try:
            exchange.create_order(
                symbol, 'limit', kapat_yon, miktar, tp, {'reduceOnly': True}
            )
        except Exception as e:
            print(f"⚠️ TP emri: {e}")
        
        AKTIF_POZISYONLAR[symbol] = {
            'yon': sig['yon'],
            'giris': giris,
            'stop': stop,
            'tp': tp,
            'miktar': miktar,
            'puan': sig['puan'],
            'giris_zaman': int(time.time() * 1000)
        }
        COIN_COOLDOWNLAR[symbol] = time.time() + 4 * 3600
        
        telegram_mesaj_gonder(
            f"🎯 *İŞLEM AÇILDI*\n"
            f"📌 `{symbol[:15]}` | *{sig['yon']}*\n"
            f"💰 Giriş: `{giris}` | SL: `{stop}` | TP: `{tp}`\n"
            f"📊 Puan: `{sig['puan']}` | Kaldıraç: `{kaldirac}x` | Kasa: `%{kasa_pct*100:.0f}`"
        )
        return True
    except Exception as e:
        print(f"❌ Pozisyon açma hatası: {e}")
        return False

# ==================== POZİSYON KAPATMA ====================
def pozisyon_kapat(symbol, sebep="MANUEL"):
    try:
        if symbol not in AKTIF_POZISYONLAR:
            return False
        
        poz = AKTIF_POZISYONLAR[symbol]
        
        try:
            for e in exchange.fetch_open_orders(symbol):
                exchange.cancel_order(e['id'], symbol)
        except Exception:
            pass
        
        kapat_yon = 'sell' if poz['yon'] == 'LONG' else 'buy'
        exchange.create_order(
            symbol, 'market', kapat_yon, poz['miktar'],
            None, {'reduceOnly': True}
        )
        
        del AKTIF_POZISYONLAR[symbol]
        telegram_mesaj_gonder(f"✅ *Kapatıldı* `{symbol[:15]}` ({sebep})")
        return True
    except Exception as e:
        print(f"❌ Kapatma hatası: {e}")
        return False

# ==================== ANA DÖNGÜ ====================
def ana_dongu():
    print(f"\n🎯 [ML BOT] Başladı | Model: {MODEL_DOSYA} | Eşik: {ML_ESIK}")
    print(f"💰 {len(COINLER)} coin izleniyor | Zaman: {ZAMAN_DILIMI}")
    
    try:
        exchange.load_markets()
        print("✅ Marketler yüklendi")
    except Exception as e:
        print(f"⚠️ Market yükleme: {e}")
    
    dongu_sayaci = 0
    while True:
        try:
            if not BOT_CALISIYOR_MU:
                time.sleep(10)
                continue
            
            su_an = time.time()
            dongu_sayaci += 1
            
            # Borsa pozisyonlarını kontrol et
            try:
                borsa_poz = exchange.fetch_positions()
                borsa_aktif = {
                    p['symbol']: p for p in borsa_poz
                    if float(p.get('contracts', 0) or 0) > 0
                }
            except Exception:
                borsa_aktif = {}
            
            # Kapananları temizle
            for sym in list(AKTIF_POZISYONLAR.keys()):
                if sym not in borsa_aktif:
                    print(f"ℹ️ {sym} kapandı, listeden silindi")
                    del AKTIF_POZISYONLAR[sym]
            
            # Yeni sinyal ara
            if len(AKTIF_POZISYONLAR) < MAKS_POZISYON:
                for coin in COINLER:
                    symbol = f'{coin}/USDT:USDT'
                    
                    if symbol in AKTIF_POZISYONLAR:
                        continue
                    if su_an < COIN_COOLDOWNLAR.get(symbol, 0):
                        continue
                    
                    df = canli_mumlari_cek(symbol, limit=200)
                    if df is None or len(df) < 100:
                        continue
                    
                    # Mum kapandı mı?
                    son_ts = int(df['timestamp'].iloc[-2])
                    if SON_MUM_TS.get(symbol) == son_ts:
                        continue
                    SON_MUM_TS[symbol] = son_ts
                    
                    df_slice = df.iloc[-100:].reset_index(drop=True)
                    sonuc = puanla(df_slice)
                    
                    if sonuc['yon'] is None or sonuc['puan'] < MIN_PUAN:
                        continue
                    
                    # ML filtresi
                    ozellikler = canli_ozellikleri_cikar(df_slice)
                    if ozellikler is None:
                        continue
                    
                    onay, proba = ml_onayla(ozellikler)
                    
                    if not onay:
                        print(f"🚫 {symbol} ML reddetti (p={proba:.3f})")
                        continue
                    
                    atr_pct = ozellikler.get('atr_pct', 1.0)
                    stop_pct = max(0.008, min(0.030, (atr_pct * 1.5) / 100.0))
                    tp_pct = stop_pct * 2.0
                    
                    sig = {
                        'yon': sonuc['yon'],
                        'puan': sonuc['puan'],
                        'fiyat': float(df_slice['close'].iloc[-1]),
                        'stop_pct': stop_pct,
                        'tp_pct': tp_pct,
                    }
                    
                    print(f"🎯 {symbol} sinyal: {sonuc['yon']} | Puan: {sonuc['puan']} | ML: {proba:.3f}")
                    
                    if pozisyon_ac(symbol, sig):
                        break
            
            # Her 10 döngüde log
            if dongu_sayaci % 10 == 0:
                print(f"🔍 #{dongu_sayaci} | Poz: {len(AKTIF_POZISYONLAR)}/{MAKS_POZISYON} | TS: {len(SON_MUM_TS)} coin")
            
            time.sleep(30)
        except Exception as e:
            print(f"⚠️ Ana döngü: {e}")
            time.sleep(30)

# ==================== TELEGRAM KOMUTLARI ====================
async def durum_komutu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        bal = exchange.fetch_balance()
        total = float(bal['total'].get('USDT', 0))
        mesaj = f"🤖 *ML BOT*\n"
        mesaj += f"💰 Bakiye: `{total:.2f}` USDT\n"
        mesaj += f"📌 Pozisyon: `{len(AKTIF_POZISYONLAR)}/{MAKS_POZISYON}`\n"
        mesaj += f"🧪 Model: `{MODEL_DOSYA}` | Eşik: `{ML_ESIK}`\n\n"
        for sym, p in AKTIF_POZISYONLAR.items():
            mesaj += f"  • `{sym[:15]}` {p['yon']} | Puan: {p['puan']} | Giriş: {p['giris']}\n"
        await update.message.reply_text(mesaj, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"Hata: {e}")

async def baslat_komutu(update, context):
    global BOT_CALISIYOR_MU
    BOT_CALISIYOR_MU = True
    await update.message.reply_text("▶️ *Bot Aktif*", parse_mode='Markdown')

async def durdur_komutu(update, context):
    global BOT_CALISIYOR_MU
    BOT_CALISIYOR_MU = False
    await update.message.reply_text("⏸️ *Durduruldu*", parse_mode='Markdown')

async def kapat_komutu(update, context):
    await update.message.reply_text("🛑 *Kapatılıyor...*", parse_mode='Markdown')
    for sym in list(AKTIF_POZISYONLAR.keys()):
        pozisyon_kapat(sym, "MANUEL_KAPAT")
    await update.message.reply_text("✅ *Bitti*", parse_mode='Markdown')

# ==================== FLASK ====================
@app.route('/')
def home():
    return f"ML Bot | Poz: {len(AKTIF_POZISYONLAR)}"

@app.route('/health')
def health():
    return "OK"

def flask_calistir():
    port = int(os.environ.get("PORT", 5000))
    try:
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    except Exception as e:
        print(f"⚠️ Flask: {e}")

# ==================== BAŞLAT ====================
if __name__ == '__main__':
    db.tablolari_olustur()
    
    threading.Thread(target=ana_dongu, daemon=True).start()
    threading.Thread(target=flask_calistir, daemon=True).start()
    
    if not TELEGRAM_TOKEN:
        print("⚠️ TELEGRAM_TOKEN yok, konsol modunda çalışıyor")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("Kapatıldı.")
    else:
        app_tg = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        app_tg.add_handler(CommandHandler("durum", durum_komutu))
        app_tg.add_handler(CommandHandler("baslat", baslat_komutu))
        app_tg.add_handler(CommandHandler("durdur", durdur_komutu))
        app_tg.add_handler(CommandHandler("kapat", kapat_komutu))
        
        print("🤖 Telegram bot başlatılıyor...")
        app_tg.run_polling(drop_pending_updates=True)
