from tqdm import tqdm
import torch

def sequence_length_penalty(length, alpha):
    return ((5 + length) / (5 + 1)) ** alpha


def BeamSearchTranslate(PAD, BOS, EOS, max_len, idx2tgt, pos_enc,
              data_loader, transformer, device):

    # Beam size and penalty alpha
    beam_size = 4
    alpha = 0.7

    pbar = tqdm(data_loader, ascii=True)
    sentence_list = []

    transformer.eval()
    with torch.no_grad():
        for batch in pbar:
            encoder_input, _, _ = map(lambda x: x.to(device), batch)
            batch_size = encoder_input.size(0) # encoder_input [batch_size, src_len]
            
            # eos = torch.full((batch_size, 1), EOS, dtype=torch.int64, device=device)
            
            src_sent_len = encoder_input.size(1)
            src_pad_masks = encoder_input.eq(PAD).unsqueeze(1) # [batch_size, 1, src_sent_len]
            
            enc_self_attn_mask = src_pad_masks.expand(-1, src_sent_len, -1)  # [batch_size, src_sent_len, src_sent_len]
            
            enc_pos_enc = pos_enc[:src_sent_len, :].unsqueeze(0)    # [1, src_sent_len, d_model]
            enc_pos_enc = enc_pos_enc.expand(batch_size, -1, -1).to(device) # [batcn_size, src_sent_len, d_model]

            encoder_outputs = transformer.encoder(encoder_input, enc_pos_enc, enc_self_attn_mask)
            # encoder_output [batcn_size, src_sent_len, d_model]

            for encoder_output, src_pad_mask in zip(encoder_outputs, src_pad_masks):
                batch_size = 1
                encoder_output = encoder_output.unsqueeze(0)
                src_pad_mask = src_pad_mask.unsqueeze(0)

                decoder_input = torch.full((batch_size, 1), BOS, dtype=torch.int64, device=device)

                max_output_length = encoder_input.shape[-1] + 50

                scores = torch.Tensor([0.]).to(device)
                vocab_size = len(idx2tgt)

                for i in range(max_output_length):
                    sent_num = decoder_input.size(0)
                    tgt_sent_len = decoder_input.size(1)
                    tgt_pad_mask = decoder_input.eq(PAD).unsqueeze(1)
                    
                    dec_self_attn_mask = tgt_pad_mask.expand(-1, tgt_sent_len, -1)  # [sent_num, tgt_sent_len, tgt_sent_len]
                    infer_mask = torch.ones((tgt_sent_len, tgt_sent_len), dtype=torch.uint8, device=device).triu(diagonal=1)
                    infer_mask = infer_mask.unsqueeze(0).expand(sent_num, -1, -1)   # [sent_num, tgt_sent_len, tgt_sent_len]
                    
                    dec_self_attn_mask = torch.gt(dec_self_attn_mask + infer_mask, 0)   # [sent_num, tgt_sent_len, tgt_sent_len]
                    dec_src_tgt_mask = src_pad_mask.expand(-1, tgt_sent_len, -1)    # [batch_size, tgt_sent_len, src_sent_len]
                
                    dec_pos_enc = pos_enc[:tgt_sent_len, :].unsqueeze(0)
                    dec_pos_enc = dec_pos_enc.expand(sent_num, -1, -1).to(device)   # [sent_num, tgt_sent_len, d_model]

                    dec_src_tgt_mask = dec_src_tgt_mask.expand(encoder_output.size(0), -1, -1)

                    # Decoder prediction
                    logits = transformer.decoder(decoder_input, encoder_output, dec_pos_enc,
                                                dec_self_attn_mask, dec_src_tgt_mask)
                    # logits [batch_size, tgt_len, dict_size]

                    # Softmax
                    log_probs = torch.log_softmax(logits[:, -1], dim=1)
                    log_probs = log_probs / sequence_length_penalty(i+1, alpha)

                    # Set score to zero where EOS has been reached
                    log_probs[decoder_input[:, -1]==EOS, :] = 0

                    # scores [beam_size, 1], log_probs [beam_size, vocab_size]
                    scores = scores.unsqueeze(1) + log_probs

                    # Flatten scores from [beams, vocab_size] to [beams * vocab_size] to get top k, 
                    # and reconstruct beam indices and token indices
                    scores, indices = torch.topk(scores.reshape(-1), beam_size)
                    beam_indices  = torch.divide(indices, vocab_size, rounding_mode='floor') # indices // vocab_size
                    token_indices = torch.remainder(indices, vocab_size) # indices %  vocab_size

                    # Build the next decoder input
                    next_decoder_input = []
                    for beam_index, token_index in zip(beam_indices, token_indices):
                        prev_decoder_input = decoder_input[beam_index]
                        if prev_decoder_input[-1]==EOS:
                            token_index = EOS # once EOS, always EOS
                        token_index = torch.LongTensor([token_index]).to(device)
                        next_decoder_input.append(torch.cat([prev_decoder_input, token_index]))
                    decoder_input = torch.vstack(next_decoder_input)

                    # If all beams are finished, exit
                    if (decoder_input[:, -1]==EOS).sum() == beam_size:
                        break

                    # Encoder output expansion from the second time step to the beam size
                    if i==0:
                        encoder_output = encoder_output.expand(beam_size, *encoder_output.shape[1:])
                    
                # convert the top scored sequence to a list of text tokens
                decoder_output, _ = max(zip(decoder_input, scores), key=lambda x: x[1])
                decoder_output = decoder_output[1:].cpu().numpy() # remove BOS

                output_text_tokens = [idx2tgt[i] for i in decoder_output if i != EOS] # remove EOS if exists
                sentence_list.append(output_text_tokens)
            
    return sentence_list

    