import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) /duration *t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return (( x +1 ) / 2 *255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps = len(images) / duration ,verbose=False)

def normalize_adjacency_matrix(adj_matrix):
    # Compute the degree matrix
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))

    # Compute the inverse of the degree matrix
    inverse_degree_matrix = np.linalg.inv(degree_matrix)

    # Compute the symmetrically normalized adjacency matrix
    sqrt_inverse_degree_matrix = np.sqrt(inverse_degree_matrix)
    normalized_adj_matrix = np.dot(np.dot(sqrt_inverse_degree_matrix, adj_matrix), sqrt_inverse_degree_matrix)

    return normalized_adj_matrix

def preprocess_wall(wall, num=1):
    wall_adj = normalize_adjacency_matrix(wall+np.identity(wall.shape[0]))
    wall_adj = np.reshape(wall_adj,[1,wall_adj.shape[0], wall_adj.shape[1]])
    wall_feat = np.ones([1, wall.shape[0],wall.shape[0]])
    wall_adj = np.repeat(wall_adj, num, axis=0)
    wall_feat = np.repeat(wall_feat, num, axis=0)

    return wall_feat, wall_adj # batchsize x wall0 x wall1

# def set_image_bandit(values ,probs ,selection ,trial):
#     bandit_image = Image.open('./resources/bandit.png')
#     draw = ImageDraw.Draw(bandit_image)
#     font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
#     draw.text((40, 10) ,str(float("{0:.2f}".format(probs[0]))) ,(0 ,0 ,0) ,font=font)
#     draw.text((130, 10) ,str(float("{0:.2f}".format(probs[1]))) ,(0 ,0 ,0) ,font=font)
#     draw.text((60, 370) ,'Trial: ' + str(trial) ,(0 ,0 ,0) ,font=font)
#     bandit_image = np.array(bandit_image)
#     bandit_image[115:115 +math.floor(values[0] *2.5),20:75,:] = [0 ,255.0 ,0]
#     bandit_image[115:115 +math.floor(values[1] *2.5) ,120:175 ,:] = [0 ,255.0 ,0]
#     bandit_image[101:107 ,10 +(selection *95):10 +(selection *95 ) +80 ,:] = [80.0 ,80.0 ,225.0]
#     return bandit_image
#
#
# def set_image_context(correct, observation ,values ,selection ,trial):
#     obs = observation * 225.0
#     obs_a = obs[: ,0:1 ,:]
#     obs_b = obs[: ,1:2 ,:]
#     cor = correct * 225.0
#     # obs_a = scipy.misc.imresize(obs_a,[100,100],interp='nearest')
#     # obs_b = scipy.misc.imresize(obs_b,[100,100],interp='nearest')
#     # cor = scipy.misc.imresize(cor,[100,100],interp='nearest')
#     obs_a = np.resize(obs_a ,np.array([100 ,100]))
#     # obs_a = obs_a.astype(np.float)/255
#
#     obs_b = np.resize(obs_b ,np.array([100 ,100]))
#     # obs_b = obs_b.astype(np.float)/255
#
#     cor = np.resize(cor ,np.array([100 ,100]))
#     # cor = cor.astype(np.float)/255
#
#     bandit_image = Image.open('./resources/c_bandit.png')
#     draw = ImageDraw.Draw(bandit_image)
#     font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
#     draw.text((50, 360) ,'Trial: ' + str(trial) ,(0 ,0 ,0) ,font=font)
#     draw.text((50, 330) ,'Reward: ' + str(values) ,(0 ,0 ,0) ,font=font)
#     bandit_image = np.array(bandit_image)
#     bandit_image[120:220 ,0:100 ,:] = obs_a
#     bandit_image[120:220 ,100:200 ,:] = obs_b
#     bandit_image[0:100 ,50:150 ,:] = cor
#     bandit_image[291:297 ,10 +(selection *95):10 +(selection *95 ) +80,:] = [80.0 ,80.0 ,225.0]
#     return bandit_image
#
#
# def set_image_gridworld(frame ,color ,reward ,step):
#     # a = scipy.misc.imresize(frame,[200,200],interp='nearest')
#     a = np.array(Image.fromarray(frame).resize([200 ,200]))
#     # print(np.unique(a))
#     b = np.ones([400 ,200 ,3]) * 255.0
#     b[0:200 ,0:200 ,:] = a
#     b[200:210 ,0:200 ,:] = np.array(color) * 255.0
#     b = Image.fromarray(b.astype('uint8'))
#     draw = ImageDraw.Draw(b)
#     font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
#     draw.text((40, 280) ,'Step: ' + str(step) ,(0 ,0 ,0) ,font=font)
#     draw.text((40, 330) ,'Reward: ' + str(reward) ,(0 ,0 ,0) ,font=font)
#     c = np.array(b)
#     return c

def set_image_mazeworld(frame ,reward ,step, ep):
    # a = scipy.misc.imresize(frame,[200,200],interp='nearest')
    a = frame * 255.0
    # print(np.unique(a))
    b = np.ones([600 ,250 ,3]) * 255.0
    b[0:250 ,0:250 ,:] = a
    b = Image.fromarray(b.astype('uint8'))
    draw = ImageDraw.Draw(b)
    font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
    draw.text((40, 280) ,'Step: ' + str(step) ,(0 ,0 ,0) ,font=font)
    draw.text((40, 330) ,'Reward: ',(0 ,0 ,0) ,font=font)
    draw.text((40, 380) ,str(reward) ,(0 ,0 ,0) ,font=font)

    name = f"run{ep}/frame{step}.jpg"
    # b.save(name)
    c = np.array(b)
    return c



def cal_discounted_reward(rewards, gamma=0.8):
    discounted_reward = []
    cumulative_sum = 0
    for i, r in enumerate(reversed(rewards)):
        cumulative_sum = (cumulative_sum + r[0][0])*gamma
        discounted_reward.append(cumulative_sum)
    return discounted_reward[::-1]