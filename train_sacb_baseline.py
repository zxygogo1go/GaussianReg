"""Train the original SACB-Net baseline with the controlled HNTS-MRG24 protocol."""

from train_registration import main


if __name__ == "__main__":
    main(expected_architecture="sacb")
