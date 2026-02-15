plugins {
    id("com.android.asset-pack")
}

assetPack {
    packName.set("models_pack")
    dynamicDelivery {
        // fast-follow: downloaded automatically right after app install
        // on-demand: downloaded only when app explicitly requests it
        deliveryType.set("fast-follow")
    }
}
