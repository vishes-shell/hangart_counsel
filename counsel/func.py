import numpy as np
import csv
from random import shuffle

from apps.shop.models import Picture


def cosine_sim(p, q):
    """
        Function that computes the similarity of two arrays without counting 0 values(Cosine similarity)
    :param p: first vector
    :type p: np.array
    :param q: second vector
    :type q: np.array
    :return: vectors' similarity
    :rtype: float
    """
    num, den_p, den_q = 0, 0, 0
    for i in range(len(p)):
        if p[i] != 0 and q[i] != 0:
            num += p[i] * q[i]
            den_p += np.square(p[i])
            den_q += np.square(q[i])
    if num != 0:
        return num / (np.sqrt(den_p) * np.sqrt(den_q))
    else:
        return 0


def similar_pics(rating_matrix):
    """
        Computes the matrix of the works' similarities between each other
    :param rating_matrix: matrix of users' rating
    :type rating_matrix: np.array
    :return: similarity matrix
    :rtype: np.array
    """
    count_pics = len(rating_matrix[0])
    sim_matrix = np.zeros(shape=(count_pics, count_pics))
    for i in range(count_pics):
        for j in range(count_pics):
            if i != j:
                sim_matrix[i][j] = cosine_sim(rating_matrix[:, i], rating_matrix[:, j])
    return sim_matrix


def fill_similar_pics(sim_matrix, picture_id_to_matrix, k=20):
    """
        Fill database with similar pics
    :param sim_matrix: similarity matrix
    :type sim_matrix: np.array
    :param picture_id_to_matrix: dictionary for matching picture_id with matrix picture index
    :type picture_id_to_matrix: dict
    :param k: number of similar pics
    :type k: int
    :return:
    """
    reverse_picture_id_to_matrix = {v: int(k) for k, v in picture_id_to_matrix.items()}
    for pics in sim_matrix:
        top_n = np.argpartition(pics, -k)[-k:]
        picture_ids = [reverse_picture_id_to_matrix[i] for i in top_n]
        Picture.similar_pictures.add(*picture_ids)


def computing_picture_id_to_matrix():
    """
        Computing dictionary for matching picture_id with matrix picture index
    :return: dictionary for matching picture_id with matrix picture index
    :rtype: dict
    """
    pictures = Picture.objects.all()
    picture_id_to_matrix = {}
    count = 0
    for picture in pictures:
        picture_id_to_matrix.update({str(picture.id): count})
        count += 1
    return picture_id_to_matrix


def get_matrix_from_csv(data_csv, test_csv, rating_matrix_npy, rating_matrix_csv, user_session_email_csv,
                        picture_id_to_matrix):
    """
        Converts csv file to matrix
    :param data_csv: path to matrix in csv
    :type data_csv: str
    :param test_csv: path to output csv file with test set for checking quality
    :type test_csv: str
    :param rating_matrix_npy: path to output npy file for rating matrix
    :type rating_matrix_npy: str
    :param rating_matrix_csv: path to output csv file for rating matrix
    :type rating_matrix_csv: str
    :param user_session_email_csv: path to output csv file for matching user matrix index with real user of session key
    :type user_session_email_csv: str
    :param picture_id_to_matrix: dictionary for matching picture_id with matrix picture index
    :type picture_id_to_matrix: dict
    :return: rating matrix
    :rtype: np.array
    """
    picture_count = Picture.objects.all().count()

    with open(data_csv, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    user_session_email = {}

    # shuffle data and split to learning and testing sets
    shuffle(data)
    count_learn_data = int(len(data) * 0.8)
    learn_data = data[:count_learn_data]
    test_data = data[count_learn_data:]

    # filling the rating matrix with learning data
    output_matrix = np.zeros((1, picture_count), dtype=np.int)
    output_matrix[0][picture_id_to_matrix[data[0][0]]] = int(learn_data[0][1])
    if learn_data[0][2] == '':
        user_session_email.update({learn_data[0][3]: 0})
    else:
        user_session_email.update({learn_data[0][2]: 0})

    for record in learn_data[1:]:
        picture_id = record[0]
        rating = int(record[1])
        session_key = record[2]
        user = record[3]

        if session_key == '':
            if user not in user_session_email:
                temp = np.zeros((1, picture_count), dtype=np.int)
                temp[0][picture_id_to_matrix[picture_id]] = rating
                output_matrix = np.append(output_matrix, temp, axis=0)
                user_session_email.update({user: len(output_matrix) - 1})
            else:
                output_matrix[user_session_email[user]][picture_id_to_matrix[picture_id]] = rating
        elif session_key not in user_session_email:
            temp = np.zeros((1, picture_count), dtype=np.int)
            temp[0][picture_id_to_matrix[picture_id]] = rating
            output_matrix = np.append(output_matrix, temp, axis=0)
            user_session_email.update({session_key: len(output_matrix) - 1})
        else:
            output_matrix[user_session_email[session_key]][picture_id_to_matrix[picture_id]] = rating

    # fill users that are consisted in test data and were not appeared in learning data
    for record in test_data:
        session_key = record[2]
        user = record[3]
        if session_key == '':
            if user not in user_session_email:
                temp = np.zeros((1, picture_count), dtype=np.int)
                output_matrix = np.append(output_matrix, temp, axis=0)
                user_session_email.update({user: len(output_matrix) - 1})
        elif session_key not in user_session_email:
            temp = np.zeros((1, picture_count), dtype=np.int)
            output_matrix = np.append(output_matrix, temp, axis=0)
            user_session_email.update({session_key: len(output_matrix) - 1})

    # save test data as csv
    with open(test_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(test_data)

    # save learning rating matrix as csv
    with open(rating_matrix_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output_matrix)

    # save user_session_email as csv
    with open(user_session_email_csv, "w", newline='') as f:
        writer = csv.writer(f)
        for key, val in user_session_email.items():
            writer.writerow([key, val])

    # save learning rating matrix as npy
    np.save(rating_matrix_npy, output_matrix)

    return output_matrix


def check_quality(test_csv, computed_data_npy, user_session_email_csv, picture_id_to_matrix):
    """
        Converts checking the quality of learned recommender system
    :param test_csv: path to csv file with test set for checking quality
    :type test_csv: str
    :param computed_data_npy: path to npy file of computed full-filled rating matrix
    :type computed_data_npy: str
    :param user_session_email_csv: path to output csv file for matching user matrix index with real user of session key
    :type user_session_email_csv: str
    :param picture_id_to_matrix: dictionary for matching picture_id with matrix picture index
    :type picture_id_to_matrix: dict
    :return: the computed RMSE
    :rtype: float
    """
    computed_data = np.load(computed_data_npy)
    with open(test_csv, 'r') as f:
        reader = csv.reader(f)
        test_data = list(reader)
    user_session_email = {}
    with open(user_session_email_csv, 'r') as f:
        reader = csv.reader(f)
        for rows in reader:
            user_session_email.update({rows[0]: int(rows[1])})

    rec_quality = 0
    for record in test_data:
        picture_id = record[0]
        rating = int(record[1])
        session_key = record[2]
        user = record[3]
        if session_key == '':
            predicted_rating = computed_data[user_session_email[user]][picture_id_to_matrix[picture_id]]
        else:
            predicted_rating = computed_data[user_session_email[session_key]][picture_id_to_matrix[picture_id]]
        rec_quality += pow(predicted_rating - rating, 2)
    rec_quality = (rec_quality / len(test_data)) ** 0.5
    return rec_quality


def get_vectors(rating_matrix, c_vector, p_vector, k_factors, steps=5000, alpha=0.002, beta=0.04):
    """
        Getting vectors of user and picture from whom the full-filled rating matrix will be computed
    :param rating_matrix: rating matrix
    :type rating_matrix: np.array
    :param c_vector: user's vector
    :type c_vector: np.array
    :param p_vector:  picture's vector
    :type p_vector: np.array
    :param k_factors: number of factors or len of c_vector and p_vector
    :type k_factors: int
    :param steps: number of iterations in optimization
    :type steps: int
    :param alpha: step for moving to minimum
    :type alpha: float
    :param beta: this is parameter is equal to lambda in the work (the reason why 'beta' is used is that 'lambda' is
                 reserved name in python), needed for regularization
    :type beta: float
    :return: vectors of user and picture
    :rtype: tuple
    """
    p_vector = p_vector.T
    for step in range(steps):
        print(step)
        for i in range(len(rating_matrix)):
            for j in range(len(rating_matrix[i])):
                if rating_matrix[i][j] > 0:
                    e_ij = rating_matrix[i][j] - np.dot(c_vector[i, :], p_vector[:, j])
                    for k in range(k_factors):
                        c_vector[i][k] += alpha * 2 * (e_ij * p_vector[k][j] - beta * c_vector[i][k])
                        p_vector[k][j] += alpha * 2 * (e_ij * c_vector[i][k] - beta * p_vector[k][j])
        e = 0
        for i in range(len(rating_matrix)):
            for j in range(len(rating_matrix[i])):
                if rating_matrix[i][j] > 0:
                    e += pow(rating_matrix[i][j] - np.dot(c_vector[i, :], p_vector[:, j]), 2)
                    for k in range(k_factors):
                        e += beta * (pow(c_vector[i][k], 2) + pow(p_vector[k][j], 2))
        if e < 0.001:
            break
    return c_vector, p_vector.T


def compute_rating_matrix(rating_matrix, final_matrix_csv, final_matrix_npy, k):
    """
        Computes the full-filled rating matrix
    :param rating_matrix: rating matrix
    :type rating_matrix: np.array
    :param final_matrix_csv: path to output csv file for filled rating matrix
    :type final_matrix_csv: str
    :param final_matrix_npy: path to output npy file for filled rating matrix
    :type final_matrix_npy: str
    :param k: number of dimensions in user's and picture's vectors
    :type k: int
    :return: full-filled rating matrix
    :rtype: np.array
    """
    n = len(rating_matrix)
    m = len(rating_matrix[0])

    c_vector = np.random.random((n, k))
    p_vector = np.random.random((m, k))
    pred_c_vector, pred_p_vector = get_vectors(rating_matrix=rating_matrix,
                                               c_vector=c_vector,
                                               p_vector=p_vector,
                                               k_factors=k)
    final_matrix = np.dot(pred_c_vector, pred_p_vector.T)

    # save final matrix as csv
    with open(final_matrix_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(final_matrix)

    # save final matrix as npy
    np.save(final_matrix_npy, final_matrix)

    return final_matrix


def similar_csv_to_db(path_to_csv, picture_id_to_matrix):
    with open(path_to_csv, mode='r') as f:
        reader = csv.reader(f)
        data = list(reader)

    reverse_picture_id_to_matrix = {v: int(k) for k, v in picture_id_to_matrix.items()}
    for idx, val in enumerate(data):
        ids = list(map(int, val))
        pic = Picture.objects.get(id=reverse_picture_id_to_matrix[idx])
        pic.similar_pictures.add(*ids)