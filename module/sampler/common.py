common = {
    "beta_end" : 0.012,
    "beta_schedule" : "scaled_linear",
    "beta_start" : 0.00085,
    "clip_sample": False,
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "trained_betas": None,
    "use_karras_sigmas": False
}

eular = {
    "num_train_timesteps": 1000,
    "beta_end" : 0.012,
    "beta_schedule" : "scaled_linear",
    "beta_start" : 0.00085,
    "trained_betas": None,
    "prediction_type": "epsilon",
    "interpolation_type": "linear",
    "use_karras_sigmas": False,
    "timestep_spacing": "leading",
    "steps_offset": 1,
    "rescale_betas_zero_snr": False,
}

eular_a = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "rescale_betas_zero_snr": False,
    "timestep_spacing": "leading",
    "steps_offset": 1,
    "trained_betas": None
}

heun = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "trained_betas": None,
    "clip_sample": False,
    "use_karras_sigmas": False,
    "timestep_spacing": "leading",
    "steps_offset": 1
}

dpm_2 = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "trained_betas": None,
    "use_karras_sigmas": False,
    "prediction_type": "epsilon",
    "timestep_spacing": "leading",
    "steps_offset": 1
}

dpm_2_ancestral = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "trained_betas": None,
    "use_karras_sigmas": False,
    "prediction_type": "epsilon",
    "timestep_spacing": "leading",
    "steps_offset": 1 
}

lms = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "use_karras_sigmas": False
}

dpmpp_2m =  {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "use_karras_sigmas": False
}