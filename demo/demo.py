import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from .utils import *


EPS = 1e-7


class Demo():
    def __init__(self, args):
        ## configs
        self.device = 'cuda:0' if args.gpu else 'cpu'
        self.checkpoint_path = args.checkpoint
        self.detect_human_face = args.detect_human_face
        self.render_video = args.render_video
        self.output_size = args.output_size
        self.image_size = 64
        self.min_depth = 0.9
        self.max_depth = 1.1
        self.border_depth = 1.05
        self.xyz_rotation_range = 60
        self.xy_translation_range = 0.1
        self.z_translation_range = 0
        self.fov = 10  # in degrees

        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth  # (-1,1) => (min_depth,max_depth)
        self.depth_inv_rescaler = lambda d :  (d-self.min_depth) / (self.max_depth-self.min_depth)  # (min_depth,max_depth) => (0,1)

        fx = (self.image_size-1)/2/(np.tan(self.fov/2 *np.pi/180))
        fy = (self.image_size-1)/2/(np.tan(self.fov/2 *np.pi/180))
        cx = (self.image_size-1)/2
        cy = (self.image_size-1)/2
        K = [[fx, 0., cx],
             [0., fy, cy],
             [0., 0., 1.]]
        K = torch.FloatTensor(K).to(self.device)
        self.inv_K = torch.inverse(K).unsqueeze(0)
        self.K = K.unsqueeze(0)

        ## NN models
        self.netD = EDDeconv(cin=3, cout=1, nf=64, zdim=256, activation=None)
        self.netA = EDDeconv(cin=3, cout=3, nf=64, zdim=256)
        self.netL = Encoder(cin=3, cout=4, nf=32)
        self.netV = Encoder(cin=3, cout=6, nf=32)

        self.netD = self.netD.to(self.device)
        self.netA = self.netA.to(self.device)
        self.netL = self.netL.to(self.device)
        self.netV = self.netV.to(self.device)
        self.load_checkpoint()

        self.netD.eval()
        self.netA.eval()
        self.netL.eval()
        self.netV.eval()

        ## face detecter
        if self.detect_human_face:
            from facenet_pytorch import MTCNN
            self.face_detector = MTCNN(select_largest=True, device=self.device)

        ## renderer
        if self.render_video:
            from unsup3d.renderer import Renderer
            assert 'cuda' in self.device, 'A GPU device is required for rendering because the neural_renderer only has GPU implementation.'
            cfgs = {
                'device': self.device,
                'image_size': self.output_size,
                'min_depth': self.min_depth,
                'max_depth': self.max_depth,
                'fov': self.fov,
            }
            self.renderer = Renderer(cfgs)

    def load_checkpoint(self):
        print(f"Loading checkpoint from {self.checkpoint_path}")
        cp = torch.load(self.checkpoint_path, map_location=self.device)
        self.netD.load_state_dict(cp['netD'])
        self.netA.load_state_dict(cp['netA'])
        self.netL.load_state_dict(cp['netL'])
        self.netV.load_state_dict(cp['netV'])

    def depth_to_3d_grid(self, depth, inv_K=None):
        if inv_K is None:
            inv_K = self.inv_K
        b, h, w = depth.shape
        grid_2d = get_grid(b, h, w, normalize=False).to(depth.device)  # Nxhxwx2
        depth = depth.unsqueeze(-1)
        grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
        grid_3d = grid_3d.matmul(inv_K.transpose(2,1)) * depth
        return grid_3d

    def get_normal_from_depth(self, depth):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth)

        tu = grid_3d[:,1:-1,2:] - grid_3d[:,1:-1,:-2]
        tv = grid_3d[:,2:,1:-1] - grid_3d[:,:-2,1:-1]
        normal = tu.cross(tv, dim=3)

        zero = normal.new_tensor([0,0,1])
        normal = torch.cat([zero.repeat(b,h-2,1,1), normal, zero.repeat(b,h-2,1,1)], 2)
        normal = torch.cat([zero.repeat(b,1,w,1), normal, zero.repeat(b,1,w,1)], 1)
        normal = normal / (((normal**2).sum(3, keepdim=True))**0.5 + EPS)
        return normal

    def detect_face(self, im):
        print("Detecting face using MTCNN face detector")
        try:
            bboxes, prob = self.face_detector.detect(im)
            w0, h0, w1, h1 = bboxes[0]
        except:
            print("Could not detect faces in the image")
            return None

        hc, wc = (h0+h1)/2, (w0+w1)/2
        crop = int(((h1-h0) + (w1-w0)) /2/2 *1.1)
        im = np.pad(im, ((crop,crop),(crop,crop),(0,0)), mode='edge')  # allow cropping outside by replicating borders
        h0 = int(hc-crop+crop + crop*0.15)
        w0 = int(wc-crop+crop)
        return im[h0:h0+crop*2, w0:w0+crop*2]

    def run(self, pil_im):
        im = np.uint8(pil_im)

        ## face detection
        if self.detect_human_face:
            im = self.detect_face(im)
            if im is None:
                return -1

        h, w, _ = im.shape
        im = torch.FloatTensor(im /255.).permute(2,0,1).unsqueeze(0)
        # resize to 128 first if too large, to avoid bilinear downsampling artifacts
        if h > self.image_size*4 and w > self.image_size*4:
            im = nn.functional.interpolate(im, (self.image_size*2, self.image_size*2), mode='bilinear', align_corners=False)
        im = nn.functional.interpolate(im, (self.image_size, self.image_size), mode='bilinear', align_corners=False)

        with torch.no_grad():
            self.input_im = im.to(self.device) *2.-1.
            b, c, h, w = self.input_im.shape

            ## predict canonical depth
            self.canon_depth_raw = self.netD(self.input_im).squeeze(1)  # BxHxW
            self.canon_depth = self.canon_depth_raw - self.canon_depth_raw.view(b,-1).mean(1).view(b,1,1)
            self.canon_depth = self.canon_depth.tanh()
            self.canon_depth = self.depth_rescaler(self.canon_depth)

            ## clamp border depth
            depth_border = torch.zeros(1,h,w-4).to(self.input_im.device)
            depth_border = nn.functional.pad(depth_border, (2,2), mode='constant', value=1)
            self.canon_depth = self.canon_depth*(1-depth_border) + depth_border *self.border_depth

            ## predict canonical albedo
            self.canon_albedo = self.netA(self.input_im)  # Bx3xHxW

            ## predict lighting
            canon_light = self.netL(self.input_im)  # Bx4
            self.canon_light_a = canon_light[:,:1] /2+0.5  # ambience term
            self.canon_light_b = canon_light[:,1:2] /2+0.5  # diffuse term
            canon_light_dxy = canon_light[:,2:]
            self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(b,1).to(self.input_im.device)], 1)
            self.canon_light_d = self.canon_light_d / ((self.canon_light_d**2).sum(1, keepdim=True))**0.5  # diffuse light direction

            ## shading
            self.canon_normal = self.get_normal_from_depth(self.canon_depth)
            self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
            canon_shading = self.canon_light_a.view(-1,1,1,1) + self.canon_light_b.view(-1,1,1,1)*self.canon_diffuse_shading
            self.canon_im = (self.canon_albedo/2+0.5) * canon_shading *2-1

            ## predict viewpoint transformation
            self.view = self.netV(self.input_im)
            self.view = torch.cat([
                self.view[:,:3] *np.pi/180 *self.xyz_rotation_range,
                self.view[:,3:5] *self.xy_translation_range,
                self.view[:,5:] *self.z_translation_range], 1)

            ## export to obj strings
            vertices = self.depth_to_3d_grid(self.canon_depth)  # BxHxWx3
            self.objs, self.mtls = export_to_obj_string(vertices, self.canon_normal)

            ## resize to output size
            self.canon_depth = nn.functional.interpolate(self.canon_depth.unsqueeze(1), (self.output_size, self.output_size), mode='bilinear', align_corners=False).squeeze(1)
            self.canon_normal = nn.functional.interpolate(self.canon_normal.permute(0,3,1,2), (self.output_size, self.output_size), mode='bilinear', align_corners=False).permute(0,2,3,1)
            self.canon_normal = self.canon_normal / (self.canon_normal**2).sum(3, keepdim=True)**0.5
            self.canon_diffuse_shading = nn.functional.interpolate(self.canon_diffuse_shading, (self.output_size, self.output_size), mode='bilinear', align_corners=False)
            self.canon_albedo = nn.functional.interpolate(self.canon_albedo, (self.output_size, self.output_size), mode='bilinear', align_corners=False)
            self.canon_im = nn.functional.interpolate(self.canon_im, (self.output_size, self.output_size), mode='bilinear', align_corners=False)

            if self.render_video:
                self.render_animation()

    def render_animation(self):
        print(f"Rendering video animations")
        b, h, w = self.canon_depth.shape

        ## morph from target view to canonical
        morph_frames = 15
        view_zero = torch.FloatTensor([0.15*np.pi/180*60, 0,0,0,0,0]).to(self.canon_depth.device)
        morph_s = torch.linspace(0, 1, morph_frames).to(self.canon_depth.device)
        view_morph = morph_s.view(-1,1,1) * view_zero.view(1,1,-1) + (1-morph_s.view(-1,1,1)) * self.view.unsqueeze(0)  # TxBx6

        ## yaw from canonical to both sides
        yaw_frames = 80
        yaw_rotations = np.linspace(-np.pi/2, np.pi/2, yaw_frames)
        # yaw_rotations = np.concatenate([yaw_rotations[40:], yaw_rotations[::-1], yaw_rotations[:40]], 0)

        ## whole rotation sequence
        view_after = torch.cat([view_morph, view_zero.repeat(yaw_frames, b, 1)], 0)
        yaw_rotations = np.concatenate([np.zeros(morph_frames), yaw_rotations], 0)

        def rearrange_frames(frames):
            morph_seq = frames[:, :morph_frames]
            yaw_seq = frames[:, morph_frames:]
            out_seq = torch.cat([
                morph_seq[:,:1].repeat(1,5,1,1,1),
                morph_seq,
                morph_seq[:,-1:].repeat(1,5,1,1,1),
                yaw_seq[:, yaw_frames//2:],
                yaw_seq.flip(1),
                yaw_seq[:, :yaw_frames//2],
                morph_seq[:,-1:].repeat(1,5,1,1,1),
                morph_seq.flip(1),
                morph_seq[:,:1].repeat(1,5,1,1,1),
            ], 1)
            return out_seq

        ## textureless shape
        front_light = torch.FloatTensor([0,0,1]).to(self.canon_depth.device)
        canon_shape_im = (self.canon_normal * front_light.view(1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shape_im = canon_shape_im.repeat(1,3,1,1) *0.7
        shape_animation = self.renderer.render_yaw(canon_shape_im, self.canon_depth, v_after=view_after, rotations=yaw_rotations)  # BxTxCxHxW
        self.shape_animation = rearrange_frames(shape_animation)

        ## normal map
        canon_normal_im = self.canon_normal.permute(0,3,1,2) /2+0.5
        normal_animation = self.renderer.render_yaw(canon_normal_im, self.canon_depth, v_after=view_after, rotations=yaw_rotations)  # BxTxCxHxW
        self.normal_animation = rearrange_frames(normal_animation)

        ## textured
        texture_animation = self.renderer.render_yaw(self.canon_im /2+0.5, self.canon_depth, v_after=view_after, rotations=yaw_rotations)  # BxTxCxHxW
        self.texture_animation = rearrange_frames(texture_animation)

    def save_results(self, save_dir):
        print(f"Saving results to {save_dir}")
        save_image(save_dir, self.input_im[0]/2+0.5, 'input_image')
        save_image(save_dir, self.depth_inv_rescaler(self.canon_depth)[0].repeat(3,1,1), 'canonical_depth')
        save_image(save_dir, self.canon_normal[0].permute(2,0,1)/2+0.5, 'canonical_normal')
        save_image(save_dir, self.canon_diffuse_shading[0].repeat(3,1,1), 'canonical_diffuse_shading')
        save_image(save_dir, self.canon_albedo[0]/2+0.5, 'canonical_albedo')
        save_image(save_dir, self.canon_im[0].clamp(-1,1)/2+0.5, 'canonical_image')

        with open(os.path.join(save_dir, 'result.mtl'), "w") as f:
            f.write(self.mtls[0].replace('$TXTFILE', './canonical_image.png'))
        with open(os.path.join(save_dir, 'result.obj'), "w") as f:
            f.write(self.objs[0].replace('$MTLFILE', './result.mtl'))

        if self.render_video:
            save_video(save_dir, self.shape_animation[0], 'shape_animation')
            save_video(save_dir, self.normal_animation[0], 'normal_animation')
            save_video(save_dir, self.texture_animation[0], 'texture_animation')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo configurations.')
    parser.add_argument('--input', default='./demo/images/human_face', type=str, help='Path to the directory containing input images')
    parser.add_argument('--result', default='./demo/results/human_face', type=str, help='Path to the directory for saving results')
    parser.add_argument('--checkpoint', default='./pretrained/pretrained_celeba/checkpoint030.pth', type=str, help='Path to the checkpoint file')
    parser.add_argument('--output_size', default=128, type=int, help='Output image size')
    parser.add_argument('--gpu', default=False, action='store_true', help='Enable GPU')
    parser.add_argument('--detect_human_face', default=False, action='store_true', help='Enable automatic human face detection. This does not detect cat faces.')
    parser.add_argument('--render_video', default=False, action='store_true', help='Render 3D animations to video')
    args = parser.parse_args()

    input_dir = args.input
    result_dir = args.result
    model = Demo(args)
    im_list = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if is_image_file(f)]

    for im_path in im_list:
        print(f"Processing {im_path}")
        pil_im = Image.open(im_path).convert('RGB')
        result_code = model.run(pil_im)
        if result_code == -1:
            print(f"Failed! Skipping {im_path}")
            continue

        save_dir = os.path.join(result_dir, os.path.splitext(os.path.basename(im_path))[0])
        model.save_results(save_dir)
