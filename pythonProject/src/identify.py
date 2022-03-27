from src import compare
import numpy as np

def similarity(dist):
    return (2-dist)*50

def find_face(test_image, db_images):
    results = {}
    data = list(db_images.values())
    data.insert(0, test_image)
    images = compare.detect_faces(data)

    for i, key in enumerate(db_images.keys(), start=1):
        pair = [images[0], images[i]]
        emb = compare.calculate_embeddings(pair)
        dist = compare.calculate_distance(emb)
        results[key] = similarity(dist)

    final = {}
    final["inputVector"] = ','.join(str(e) for e in emb[0, :])
    final["analysis"] = dict(sorted(results.items(), key=lambda item: item[1]))

    return final

def calculate_embeddings(inputs):
    results = {}
    images = compare.detect_faces(list(inputs.values()))

    for i, key in enumerate(inputs.keys()):
        results[key] = compare.calculate_embeddings([images[i]])[0]
    return results

def calculate_distance(test_image, db_embeddings):
    db = {}
    results = {}
    input = compare.detect_faces([test_image])[0]
    input_embedding = compare.calculate_embeddings([input])[0]

    for key in db_embeddings.keys():
        db[key] = list(map(float, db_embeddings[key].split(',')))
        dist = compare.calculate_distance(np.array([input_embedding, db[key]]))
        results[key] = similarity(dist)

    final = {}
    final["inputVector"] = ','.join(str(e) for e in input_embedding)
    final["analysis"] = dict(sorted(results.items(), key=lambda item: item[1]))

    return final
