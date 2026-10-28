"""Local occupancy generation server backed by the standalone ss_module package."""
import argparse
import json
import logging
import os
import sys
import threading
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError
from plotly.offline import get_plotlyjs

app = Flask(__name__, template_folder=str(Path(__file__).parent / 'templates'),
            static_folder=str(Path(__file__).parent / 'static'))
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
OUTPUTS = Path(__file__).parent / 'outputs'
jobs = {}
lock = threading.Lock()
pipeline = None


def load_pipeline():
    global pipeline
    if pipeline is None:
        from ss_module.pipelines import OccupancyPipeline
        weights = os.environ.get('SS_WEIGHTS', str(ROOT / 'weights'))
        decoder = os.environ.get('SS_DECODER_WEIGHTS')
        pipeline = OccupancyPipeline(weights, decoder_weights=decoder)
    return pipeline


def generate(job_id, images, params):
    job = jobs[job_id]
    try:
        job.update(status='running', message='正在加载模型，首次运行可能需要下载 DINOv2…')
        pipe = load_pipeline()
        job['message'] = '正在去除背景并编码图片…'
        def progress(current, total):
            job['message'] = f'正在采样稀疏结构：{current}/{total} 步…'
        result = pipe.generate(images=images, steps=params['steps'], seed=params['seed'],
                               cfg_strength=params['cfg'], mode=params['mode'], progress=progress)
        occupancy, coords, logits = result['occupancy'], result['coords'], result['logits']
        directory = OUTPUTS / job_id
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / 'occupancy.npz', occupancy=occupancy,
                            coords=coords, logits=logits)
        job['message'] = '正在构建体素中心点与边线预览…'
        centers = ((coords.astype(float) + 0.5) / 64 - 0.5).tolist()
        (directory / 'preview.json').write_text(json.dumps(
            {'centers': centers}))
        job.update(count=len(coords), status='done', message='生成完成')
    except Exception:
        logging.exception('Occupancy generation failed')
        job.update(status='error', message='生成失败，请查看终端日志。可尝试减少图片数或采样步数。')
    finally:
        lock.release()


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/plotly.js')
def plotly_js():
    return app.response_class(get_plotlyjs(), mimetype='application/javascript')


@app.post('/api/generate')
def start():
    files = request.files.getlist('images')
    if not 1 <= len(files) <= 12:
        return jsonify(error='请选择 1～12 张同一物体的图片。'), 400
    try:
        params = dict(steps=int(request.form.get('steps', 12)),
                      seed=int(request.form.get('seed', 1)),
                      cfg=float(request.form.get('cfg', 7.5)),
                      mode=request.form.get('mode', 'stochastic'))
        if not (1 <= params['steps'] <= 100 and 0 <= params['seed'] < 2**32
                and 0 <= params['cfg'] <= 30 and params['mode'] in ('stochastic', 'multidiffusion')):
            raise ValueError()
        if params['mode'] == 'stochastic' and len(files) > params['steps']:
            return jsonify(error='轮流模式下，采样步数不能少于图片数量。'), 400
        images = []
        for file in files:
            with Image.open(file.stream) as image:
                if image.width * image.height > 25_000_000:
                    raise ValueError()
                image.load()
                images.append(ImageOps.exif_transpose(image).convert('RGBA'))
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
        return jsonify(error='图片无法读取或参数无效；每张图片请小于 2500 万像素。'), 400
    if not lock.acquire(blocking=False):
        return jsonify(error='当前有生成任务正在运行，请稍后重试。'), 409
    job_id = uuid.uuid4().hex
    jobs[job_id] = dict(status='queued', message='任务已提交')
    threading.Thread(target=generate, args=(job_id, images, params), daemon=True).start()
    return jsonify(id=job_id), 202


@app.get('/api/jobs/<job_id>')
def status(job_id):
    if job_id not in jobs:
        return jsonify(error='任务不存在'), 404
    return jsonify(jobs[job_id])


@app.get('/api/jobs/<job_id>/<kind>')
def artifact(job_id, kind):
    if jobs.get(job_id, {}).get('status') != 'done' or kind not in ('preview', 'download'):
        return jsonify(error='结果不存在或尚未完成'), 404
    filename = 'preview.json' if kind == 'preview' else 'occupancy.npz'
    return send_file(OUTPUTS / job_id / filename, as_attachment=kind == 'download')


@app.errorhandler(413)
def too_large(error):
    return jsonify(error='上传总大小不能超过 32 MB。'), 413


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
