


def train(train_queue, model, cnn_optimizer, grad_scalar, global_step, warmup_iters, writer, logging):
    alpha_i = utils.kl_balancer_coeff(num_scales=model.num_latent_scales,
                                      groups_per_scale=model.groups_per_scale, fun='square')
    nelbo = utils.AvgrageMeter()
    model.train()
    for step, x in enumerate(train_queue):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()

        # change bit length
        x = utils.pre_process(x, args.num_x_bits)

        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = lr

        # sync parameters, it may not be necessary
        if step % 100 == 0:
            utils.average_params(model.parameters(), args.distributed)

        cnn_optimizer.zero_grad()
        with autocast():
            logits, log_q, log_p, kl_all, kl_diag = model(x)

            output = model.decoder_output(logits)
            kl_coeff = utils.kl_coeff(global_step, args.kl_anneal_portion * args.num_total_iter,
                                      args.kl_const_portion * args.num_total_iter, args.kl_const_coeff)

            recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
            
            balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)

            nelbo_batch = recon_loss + balanced_kl
            loss = torch.mean(nelbo_batch)
            norm_loss = model.spectral_norm_parallel()
            bn_loss = model.batchnorm_loss()
            # get spectral regularization coefficient (lambda)
            if args.weight_decay_norm_anneal:
                assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
                wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(args.weight_decay_norm)
                wdn_coeff = np.exp(wdn_coeff)
            else:
                wdn_coeff = args.weight_decay_norm

            loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

        grad_scalar.scale(loss).backward()
        utils.average_gradients(model.parameters(), args.distributed)
        grad_scalar.step(cnn_optimizer)
        grad_scalar.update()
        nelbo.update(loss.data, 1)

        if (global_step + 1) % 100 == 0:
            if (global_step + 1) % 1000 == 0:  # reduced frequency
                n = int(np.floor(np.sqrt(x.size(0))))
                x_img = x[:n*n]
                output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.sample()
                output_img = output_img[:n*n]
                x_tiled = utils.tile_image(x_img, n)
                output_tiled = utils.tile_image(output_img, n)
                in_out_tiled = torch.cat((x_tiled, output_tiled), dim=2)
                writer.add_image('reconstruction', in_out_tiled, global_step)

            # norm
            writer.add_scalar('train/norm_loss', norm_loss, global_step)
            writer.add_scalar('train/bn_loss', bn_loss, global_step)
            writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)

            utils.average_tensor(nelbo.avg, args.distributed)
            logging.info('train %d %f', global_step, nelbo.avg)
            writer.add_scalar('train/nelbo_avg', nelbo.avg, global_step)
            writer.add_scalar('train/lr', cnn_optimizer.state_dict()[
                              'param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/nelbo_iter', loss, global_step)
            writer.add_scalar('train/kl_iter', torch.mean(sum(kl_all)), global_step)
            writer.add_scalar('train/recon_iter', torch.mean(utils.reconstruction_loss(output, x, crop=model.crop_output)), global_step)
            writer.add_scalar('kl_coeff/coeff', kl_coeff, global_step)
            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag):
                utils.average_tensor(kl_diag_i, args.distributed)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

                # kl_ceoff
                writer.add_scalar('kl/active_%d' % i, num_active, global_step)
                writer.add_scalar('kl_coeff/layer_%d' % i, kl_coeffs[i], global_step)
                writer.add_scalar('kl_vals/layer_%d' % i, kl_vals[i], global_step)
            writer.add_scalar('kl/total_active', total_active, global_step)

        global_step += 1

    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step


def test(valid_queue, model, num_samples, args, logging):
    if args.distributed:
        dist.barrier()
    nelbo_avg = utils.AvgrageMeter()
    neg_log_p_avg = utils.AvgrageMeter()
    model.eval()
    for step, x in enumerate(valid_queue):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()

        # change bit length
        x = utils.pre_process(x, args.num_x_bits)

        with torch.no_grad():
            nelbo, log_iw = [], []
            for k in range(num_samples):
                logits, log_q, log_p, kl_all, _ = model(x)
                output = model.decoder_output(logits)
                recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
                balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
                nelbo_batch = recon_loss + balanced_kl
                nelbo.append(nelbo_batch)
                log_iw.append(utils.log_iw(output, x, log_q, log_p, crop=model.crop_output))

            nelbo = torch.mean(torch.stack(nelbo, dim=1))
            log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))

        nelbo_avg.update(nelbo.data, x.size(0))
        neg_log_p_avg.update(- log_p.data, x.size(0))

    utils.average_tensor(nelbo_avg.avg, args.distributed)
    utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, NELBO: %f, neg Log p %f', step, nelbo_avg.avg, neg_log_p_avg.avg)
    return neg_log_p_avg.avg, nelbo_avg.avg